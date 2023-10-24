import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict, Counter
import numpy as np
import os

#
# helper functions
#
char_replacement = {u'é':u'e1', u'è':u'e2', u'ẽ':u'e3',
                    u'ò':u'o2', u'õ':u'o2',
                    u'ú':u'u1', u'ù':u'u2',
                    u'à':u'a2', u'ã':u'a3',
                    u'ǜ':u'ü2',
                    u'ì':u'i2',
                   }

def replace_ngraphs(s):
    for old, new in [('sch', '8'), ('ch', '9')]:
        s = s.replace(old, new)
    return s

def replaced(s):
    return ''.join(char_replacement.get(c, c) for c in replace_ngraphs(s))


#
# data encoding
#
padding='pre'
truncating='post'

def encode_data(X, encoder,
                maxlen, fit=False,
                padding=padding,
                truncating=truncating,
                padding_value=0):
    # `fit=True` for training, `fit=False` otherwise
    # for production: better intelligently replace chars not in encoder
    return pad_sequences(([[encoder[c] for c in s] for s in X] if fit else
                          # in dev and test, if char not in encoder, ignore it
                          [[encoder[c] for c in s if c in encoder] for s in X]),
                         maxlen=maxlen, dtype='int32',
                         padding=padding, truncating=truncating,
                         value=padding_value)

def decoder(enc):
    return dict((v, k) for k, v in enc.items())

def encode_store(input_str,
                 maxlen=70,
                 model_dir=False):
    # one hot encode characters
    encoder = defaultdict()
    encoder.default_factory = encoder.__len__
    padding_value = encoder['#']
    assert padding_value == 0

    # encoder data
    # prepend with a padding value, truncate long sequences at the end
    encoder = dict({"#": 0, "u": 1, "2": 2, "n": 3, "d": 4, " ": 5, "e": 6, "t": 7, "h": 8, "m": 9, "g": 10, "\u00e4": 11, "l": 12, "r": 13, "i": 14, "o": 15, "9": 16, "b": 17, "8": 18, "s": 19, "a": 20, "\u00f6": 21, "v": 22, "w": 23, "p": 24, "z": 25, "f": 26, "\u00fc": 27, "k": 28, "j": 29, "1": 30, "\u0303": 31, "\u0300": 32, "3": 33, "x": 34, "q": 35, "c": 36})
    print('# chars %d' % len(encoder))

    # replace special chars of input string
    replaced_string = replaced(input_str)

    # encode input string
    X_pred = encode_data([replaced(replaced_string)], encoder, maxlen)

    return X_pred

# one-hot encode data as dense vectors
def onehot_encode_data(X, vocab_size):
    num_samples, num_timesteps = X.shape
    X_enc = np.zeros((num_samples, num_timesteps, vocab_size),
                     dtype=np.int8)
    # how to do it with numpy indexing?
    for i in ((s, t, f - 1) for s in range(num_samples)
              for t, f in enumerate(X[s]) if f != 0):
        X_enc[i] = 1
    return X_enc



#
# Main Script
#
INPUT_STRING = "Aber zisti wär o immet no okay. Mitteuch hätti meh zyt aber"


# Encode Input String
X_pred = encode_store(input_str=INPUT_STRING)
vocab_size = 36 # X_pred.max()  # NB: 0 is not in the vocabulary, but is padding
seq_length = X_pred.shape[1]
print('Vocabulary size, sequence length: %d, %d' % (vocab_size, seq_length))
X_pred = onehot_encode_data(X_pred, vocab_size)
print('Encoded Input String.')


# Load Model
model = tf.keras.models.load_model('./lib/my_model.keras')

# Model Prediction
dialects = ['LU', 'BE', 'ZH', 'BS']
softmax_predictions = model.predict(X_pred)
prediction = np.argmax(softmax_predictions)

print(f"{dialects[prediction]} ({softmax_predictions[0][prediction]:.2%})")