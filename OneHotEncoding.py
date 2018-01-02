from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

data = ['warm', 'warm', 'cold', 'hot', 'hot', 'warm', 'warm', 'hot', 'hot', 'hot', 'cold', 'cold']
values = array(data)
print(values)

# integer encoding
encoder = LabelEncoder()
integerValues = encoder.fit_transform(values)
print(integerValues)

# one hot encoding
oneHotEncoder = OneHotEncoder(sparse=False)
integerValues = integerValues.reshape(len(integerValues), 1)
oneHotEncodedValues = oneHotEncoder.fit_transform(integerValues)
print(oneHotEncodedValues)

# invert
print(encoder.inverse_transform(argmax(oneHotEncodedValues[0, :])))
