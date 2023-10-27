#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json

import numpy as np

np.random.seed(42)

from keras.models import Sequential
from keras.layers import Dense, LSTM, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import keras_tuner as kt
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


gpus = ['0'] #'4', '5', '6', '7']
os.environ['CUDA_VISIBLE_DEVICES'] = (",").join(gpus)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
config = tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
tf_session = tf.compat.v1.Session(config=config)  # noqa: F841


# select data model, we use "model+charrep+augm"
data_model = sys.argv[1]
print('Using data model "%s".' % data_model)

# define batch size and directories
batch_size = 32
data_dir = 'preprocessed_data.d/paper_runs/'
models_dir = 'lstm_models.d/paper_runs'


# one-hot encode data as dense vectors
def encode_data(X, vocab_size):
    num_samples, num_timesteps = X.shape
    X_enc = np.zeros((num_samples, num_timesteps, vocab_size),
                     dtype=np.int8)
    # how to do it with numpy indexing?
    for i in ((s, t, f - 1) for s in range(num_samples)
              for t, f in enumerate(X[s]) if f != 0):
        X_enc[i] = 1
    return X_enc


def load_data(data_model, fn):
    return np.loadtxt(
        os.path.join(data_dir, data_model, fn + '.txt'),
        dtype=np.dtype('int32'))


# load and encode data
X_train = load_data(data_model, 'X_train')
vocab_size = X_train.max()  # NB: 0 is not in the vocabulary, but is padding
seq_length = X_train.shape[1]
print('Vocabulary size, sequence length: %d, %d' % (vocab_size, seq_length))

X_train = encode_data(X_train, vocab_size)
X_dev = encode_data(load_data(data_model, 'X_dev'), vocab_size)
X_test = encode_data(load_data(data_model, 'X_test'), vocab_size)

y_train = load_data(data_model, 'y_train')
y_dev = load_data(data_model, 'y_dev')
y_test = load_data(data_model, 'y_test')
print('Loaded data.')

# define a hypermodel for tuning
def build_model(hp):
    model = Sequential()
    model.add(Masking(mask_value=0.,
                      input_shape=(seq_length, vocab_size),
                      name='Masking'))

    lstm_layers = hp.Choice('num_lstm_layers', values=[1])

    model.add(LSTM(units=hp.Int(f'lstm_units_layer_0', min_value=32, max_value=256, step=32),
                    recurrent_dropout=hp.Float(f'recurrent_dropout_layer_0', min_value=0.1, max_value=0.5,
                                                step=0.1),
                    dropout=hp.Float(f'dropout_layer_0', min_value=0.1, max_value=0.5, step=0.1),
                    name=f'LSTM_layer_0'))

    model.add(Dense(units=4,
                    activation='softmax',
                    name='Softmax'))

    # Choose between Adam and RMSprop
    optimizer_choice = hp.Choice('optimizer', values=['adam', 'rmsprop'])
    if optimizer_choice == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]))
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


# overload build_model function as a way to manually specify hyperparameters
def build_model(num_lstm_layers, lstm_units_layer_0, recurrent_dropout_layer_0, dropout_layer_0, optimizer, learning_rate):
    model = Sequential()
    model.add(Masking(mask_value=0.,
                      input_shape=(seq_length, vocab_size),
                      name='Masking'))

    model.add(LSTM(units=lstm_units_layer_0,
                    recurrent_dropout=recurrent_dropout_layer_0,
                    dropout=dropout_layer_0,
                    name=f'LSTM_layer_0'))

    model.add(Dense(units=4,
                    activation='softmax',
                    name='Softmax'))

    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy', 'categorical_accuracy'])

    return model


# define model checkpoint callback
filepath = 'best_model.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')


# build the model with the fixed hyperparameters
model = build_model(**fixed_hyperparameters)

# train the model
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=100,  # Adjust as needed
          validation_data=(X_dev, y_dev),
          verbose=1,
          callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=10), checkpoint])

#
# Evaluation
#

# Load best model
model = tf.keras.models.load_model('best_model.hdf5')
print(f"Amount of samples during training: {X_train.shape[0]}")

# Dev Set Evaluation
test_loss, test_accuracy, test_categorical_accuracy = model.evaluate(X_dev, y_dev, batch_size=batch_size)
print(f'Dev Loss: {test_loss}')
print(f'Dev Accuracy: {test_accuracy:.2%}')

# F1 scores
y_pred = model.predict(X_dev)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_dev, axis=1)

# Calculate F1 score
f1_m = f1_score(y_true, y_pred, average='macro')
f1_w = f1_score(y_true, y_pred, average='weighted')

print(f'Dev F1 Score (macro): {f1_m:.2%}')
print(f'Dev F1 Score (weighted): {f1_w:.2%}')


# Test Set Evaluation

# Evaluate the model on the test set
test_loss, test_accuracy, test_categorical_accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy:.2%}')


# F1 scores
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculate F1 score
f1_m = f1_score(y_true, y_pred, average='macro')
f1_w = f1_score(y_true, y_pred, average='weighted')

print(f'Test F1 Score (macro): {f1_m:.2%}')
print(f'Test F1 Score (weighted): {f1_w:.2%}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['LU', 'BE', 'ZH', 'BS'], yticklabels=['LU', 'BE', 'ZH', 'BS'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Test Set Confusion Matrix')

# Save the plot
plt.savefig('confusion_matrix.png')
plt.show()