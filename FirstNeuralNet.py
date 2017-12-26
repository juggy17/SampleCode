from keras.layers import Dense
from keras.models import Sequential
import numpy
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.callbacks import ModelCheckpoint

def CreateModel(optimizer='rmsprop', init='glorot_uniform'):
    # model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# random seed
seed = 8
numpy.random.seed(seed)

# load txt file
dataset = numpy.loadtxt("../code/chapter_07/pima-indians-diabetes.csv", delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]

# split into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

# create model
# model = KerasClassifier(build_fn=CreateModel, verbose=0)
#
# # optimizers
# optimizers = ['adam', 'rmsprop']
# inits = ['glorot_uniform', 'normal', 'uniform']
# epochs = [50, 100, 150]
# batches = [5, 10, 20]
#
# param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=inits)
# grid = GridSearchCV(estimator=model, param_grid=param_grid)
#
# grid_result = grid.fit(X, Y)
#
#
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

# compile
model = CreateModel()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# checkpoint
filepath = "weights.best.hd5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='Max')
callbacks_List = [checkpoint]
# fit
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=150, batch_size=20, callbacks=callbacks_List)

# evaluate
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
