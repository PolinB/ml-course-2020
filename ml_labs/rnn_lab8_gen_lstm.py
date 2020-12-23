import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.utils import np_utils
import re

# filename = "lab8_files/saltan.txt"
# filename = "lab8_files/format-robinzon.txt"
# filename = "lab8_files/wonderland.txt"
filename = "lab8_files/format_wond.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
raw_text = re.sub('\n+', '\n', raw_text)

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)
# print("Total Characters: ", n_chars)
# print("Size alphabet: ", n_vocab)

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
# model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

# filename = "lab8_files/saltan/weights-improvement-30-2.2312.hdf5"
# filename = "lab8_files/saltan2/weights-improvement-30-1.5444.hdf5"
# filename = "lab8_files/robinzon2/weights-improvement-20-2.0075.hdf5"
# filename = "lab8_files/wonderland/weights-improvement-10-2.3477.hdf5"
# filename = "lab8_files/wonderland2/weights-improvement-02-2.9100.hdf5"
# filename = "lab8_files/wonderland3/weights-improvement-03-2.7619.hdf5"
# filename = "lab8_files/wonderland4/weights-improvement-19-1.9738.hdf5"
filename = "lab8_files/wonderland5/weights-improvement-30-1.3317.hdf5"
model.load_weights(filename)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("Start:")
print(''.join([int_to_char[value] for value in pattern]))

print("Gen:")
for i in range(500):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone.")
