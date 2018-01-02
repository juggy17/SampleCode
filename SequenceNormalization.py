from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# define contrived series
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

series = Series(data)
print(series)
values = series.values

# make a column vector
values = values.reshape(len(values), 1)
print(values)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))

normalizedData = scaler.transform(values)
print(normalizedData)

# invert the transform
invertedData = scaler.inverse_transform(normalizedData)
print(invertedData)

# Standardization
scaler = StandardScaler()
scaler = scaler.fit(values)

standardizedData = scaler.transform(values)
print(standardizedData)

# invert
invertedData = scaler.inverse_transform(standardizedData)
print(invertedData)