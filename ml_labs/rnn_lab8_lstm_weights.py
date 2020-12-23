import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import re


def format_file(filename, outfilename):
    text = open(filename).read()
    text = text.lower()
    # text = text.replace(".", "\n")
    # text = text.replace("!", "\n")
    # text = text.replace("?", "\n")
    new_text = ""
    low = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
           'w', 'x', 'y', 'z']
    for c in text:
        if (c.isalpha() and (c in low)) or c.isspace() or c == '”' or c == '\'':
            new_text += c
    new_text = re.sub(' +', ' ', new_text)
    new_text = re.sub('\n+', '\n', new_text)
    new_text = re.sub('\n ', '\n', new_text)
    open(outfilename, "w").write(new_text)


# format_file("lab8_files/robinzon-kruzo.txt", "lab8_files/format-robinzon.txt")
# filename = "lab8_files/format-robinzon.txt"
# format_file("lab8_files/wonderland.txt", "lab8_files/format_wond.txt")
# filename = "lab8_files/format_wond.txt"
# filename = "lab8_files/wonderland.txt"
filename = "lab8_files/saltan.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
raw_text = re.sub('\n+', '\n', raw_text)

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

print(chars)

n_chars = len(raw_text)
n_vocab = len(chars)

seq_length = 50
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)


X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
# model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# filepath = "lab8_files/robinzon/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# filepath = "lab8_files/robinzon2/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
filepath = "lab8_files/saltan2/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=30, batch_size=64, callbacks=callbacks_list)

