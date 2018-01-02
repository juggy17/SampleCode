import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# generate the input sequence
def GenerateSequence(length, nFeatures):
    return np.array([np.random.randint(0, nFeatures) for _ in range(length)])

# variables
seqLength = 5
numFeatures = 10
outIndex = 2

# create model
model = Sequential()
model.add(LSTM(25, input_shape=(seqLength, numFeatures)))
model.add(Dense(numFeatures, activation='softmax'))
model.compile(loss='categoric'
                   'al_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())

# train
for epochs in range(10000):
    # create an input sequence
    x = GenerateSequence(seqLength, numFeatures).reshape(seqLength, 1)
    # one hot encoding
    oneHotEncoder = OneHotEncoder(n_values=numFeatures, sparse=False)
    oneHotEncoderValues = oneHotEncoder.fit_transform(x)

    # generate X and y for training LSTM
    X = oneHotEncoderValues.reshape(1, seqLength, numFeatures)
    y = oneHotEncoderValues[outIndex].reshape(1, numFeatures)

    model.fit(X, y, epochs=1, verbose=2)

# predict
correct = 0
for samples in range(100):
    # create an input sequence
    x = GenerateSequence(seqLength, numFeatures).reshape(seqLength, 1)
    # one hot encoding
    oneHotEncoder = OneHotEncoder(n_values=numFeatures, sparse=False)
    oneHotEncoderValues = oneHotEncoder.fit_transform(x)

    # generate X and y for training LSTM
    X = oneHotEncoderValues.reshape(1, seqLength, numFeatures)
    y = oneHotEncoderValues[outIndex].reshape(1, numFeatures)

    yhat = model.predict(X)
    if([np.argmax(val) for val in y] == [np.argmax(val) for val in yhat]):
        correct += 1

print(correct)

