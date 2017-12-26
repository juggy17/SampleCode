import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.callbacks import ModelCheckpoint

# random seed
seed = 8
numpy.random.seed(seed)

# load values
data = read_csv('../code/chapter_10/iris.csv', header=None)
data = data.values
x = data[:, 0:4].astype(float)
y = data[:, 4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# one-hot code
dummy_y = np_utils.to_categorical(encoded_y)

def base_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']   )
    return model


filepath = "weights.best.hd5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='Max')
callbacks_List = [checkpoint]

estimator = KerasClassifier(build_fn=base_model(), epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, x, dummy_y)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
