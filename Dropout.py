import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import  KerasClassifier
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.layers import Dropout

seed = 8
numpy.random.seed(seed)

dataFrame =read_csv('../code/chapter_16/sonar.csv', header=None)
dataSet = dataFrame.values

X = dataSet[:, 0:60].astype(float)
Y = dataSet[:, 60]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

# base line model
def baseline_model():
    model = Sequential()
    #model.add(Dropout(0.2, input_shape=(60,)))
    model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=baseline_model(), epochs=300, batch_size=16, verbose=0)))
pipeline = Pipeline(estimators)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
model = baseline_model()

X_train, X_test, Y_train, Y_test = train_test_split(X, encoded_Y, test_size=0.2, random_state=seed)

results = model.fit(X_train, Y_train, validation_split=0.3, batch_size=10, epochs=150, verbose=1)
results = model.evaluate(X_test, Y_test, verbose=1)
print(results)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))