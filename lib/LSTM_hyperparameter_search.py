#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# In[1]:

import os
import sys
import json

import numpy as np

np.random.seed(42)

from keras.models import Sequential
from keras.layers import Dense, LSTM, Masking
from keras.callbacks import EarlyStopping
import tensorflow as tf
import keras_tuner as kt

gpus = ['0', '1', '2', '3'] #'4', '5', '6', '7']
os.environ['CUDA_VISIBLE_DEVICES'] = (",").join(gpus)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
config = tf.compat.v1.ConfigProto(log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
tf_session = tf.compat.v1.Session(config=config)  # noqa: F841

# In[2]:

"""
one-directional LSTM model without character embedding

 * define a model and run for 400 iterations without early stopping,
 * store model params at every 100 iterations.
"""

# In[3]:

data_model = sys.argv[1]
print('Using data model "%s".' % data_model)

#lstm_size = 90
#nb_epochs = (100,) * 4
batch_size = 32

# In[49]:

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

# Define a hypermodel for tuning
def build_model(hp):
    model = Sequential()
    model.add(Masking(mask_value=0.,
                      input_shape=(seq_length, vocab_size),
                      name='Masking'))

    for i in range(hp.Choice('num_lstm_layers', values=[1, 2])):
        model.add(LSTM(units=hp.Int(f'lstm_units_layer_{i}', min_value=32, max_value=256, step=32),
                       recurrent_dropout=hp.Float(f'recurrent_dropout_layer_{i}', min_value=0.1, max_value=0.5,
                                                  step=0.1),
                       dropout=hp.Float(f'dropout_layer_{i}', min_value=0.1, max_value=0.5, step=0.1),
                       return_sequences=(i < hp.Choice('num_lstm_layers', values=[1, 2])),
                       name=f'LSTM_layer_{i}'))

    for i in range(hp.Choice('num_dense_layers', values=[0, 1])):
        model.add(Dense(units=hp.Int(f'dense_units_layer_{i}', min_value=16, max_value=128, step=16),
                        activation='relu',
                        name=f'Dense_layer_{i}'))

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

def save_model(i, model,
               data_model,
               with_emb=True,
               test_loss=None,
               test_accuracy=None
               ):
    path = os.path.join(models_dir,
                        ('emb' if with_emb else 'no_emb'),
                        data_model)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # serealize model
    model_json = model.to_json()
    with open(os.path.join(path,
                           'model_%d.json' % i), 'w') as json_file:
        json_file.write(model_json)

    # serialize weights
    model.save_weights(os.path.join(path,
                                    'model_%d.h5' % i))

    # Add test_loss and test_accuracy to model JSON
    model_info = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }
    model_info_json = json.dumps(model_info)
    with open(os.path.join(path, 'model_info_%d.json' % i), 'w') as json_file:
        json_file.write(model_info_json)


gpu_names = [f'GPU:{num}' for num in gpus]

# Define a tuner
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective('val_loss', direction='min'),
    max_epochs=100,
    factor=3,
    directory='results',
    distribution_strategy=tf.distribute.MirroredStrategy(devices=gpu_names)  # Assign GPUs to workers
)

# Search for the best hyperparameters
tuner.search(X_train, y_train,
             epochs=100,
             validation_data=(X_dev, y_dev),
             callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=8)])

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters
best_model = tuner.hypermodel.build(best_hps)

# Train the best model on the entire training set
#best_model.fit(X_train, y_train,
#               batch_size=batch_size,
#               epochs=20,  # Adjust as needed
#               validation_data=(X_dev, y_dev),
#               verbose=1)

# Evaluate the best model on the test set
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, batch_size=batch_size)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

save_model(0, best_model, data_model, with_emb=False, test_loss=test_loss, test_accuracy=test_accuracy)
