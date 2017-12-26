import numpy
from keras.datasets import imdb
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

seed = 8
numpy.random.seed(seed)

# load the dataset
topWords = 5000
maxWords = 500
numDim = 32
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=topWords)
X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)

# model
model = Sequential()
model.add(Embedding(topWords, numDim, input_length=maxWords))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

lrate = 0.01
epochs = 25
decay = lrate/epochs

sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_split=0.3, epochs=2, batch_size=128, verbose=2)
scores = model.evaluate(X_test, y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))