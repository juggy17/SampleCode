from keras.datasets import cifar10
from matplotlib import pyplot
from scipy.misc import toimage
import numpy
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.constraints import maxnorm
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')
(x_tr, y_tr), (x_te, y_te) = cifar10.load_data()
for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(toimage(x_tr[i]))

pyplot.show()



seed = 8
numpy.random.seed(seed)

x_tr = x_tr.astype('float32')
x_te = x_te.astype('float32')

x_tr = x_tr/255
x_te = x_te/255

y_tr = np_utils.to_categorical(y_tr)
y_te = np_utils.to_categorical(y_te)

def base_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    lrate = 0.01
    epochs = 25
    decay = lrate/epochs

    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


numClasses = y_tr.shape[1]

model = base_model()
model.fit(x_tr, y_tr, validation_split=0.3, batch_size=10, epochs=25)
scores = model.evaluate(x_tr, y_tr, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
