from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from matplotlib import pyplot
from math import sin, pi, exp
from random import uniform, randint
from numpy import array

# generate damped sine wave
def GenerateSequence(length, period, decay):
    return [0.5 + 0.5*sin(2*pi*i/period)*exp(-decay*i) for i in range(length)]

# generate training data
def GenerateSequences(inputLength, numSequences, outputLength):
    X, y = list(), list()

    # loop through the sequences
    for _ in range(numSequences):
        p = randint(10, 20)
        d = uniform(0.01, 0.1)
        sequence = GenerateSequence(inputLength + outputLength, p, d)
        X.append(sequence[:-outputLength])
        y.append(sequence[-outputLength:])

    X = array(X).reshape(numSequences, inputLength, 1)
    y = array(y).reshape(numSequences, outputLength)
    return X, y


# variables
inputLength = 50
outputLength = 10
numSequences = 10000
batchSize = 20

# model
model = Sequential()
model.add(LSTM(20, return_sequences=True, input_shape=(inputLength, 1)))
model.add(LSTM(20))
model.add(Dense(outputLength))
model.compile(optimizer='adam', loss='mae', metrics=['acc'])
print(model.summary())

# train
X, y = GenerateSequences(inputLength, numSequences, outputLength)
history = model.fit(X, y, batch_size=batchSize, epochs=1)

# eval
X, y = GenerateSequences(inputLength, 1, outputLength)
yhat = model.predict(X)

pyplot.plot(y[0], label='y')
pyplot.plot(yhat[0], label='yhat')
pyplot.legend()
pyplot.show()
