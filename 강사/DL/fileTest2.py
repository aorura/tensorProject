import numpy as np

data=np.loadtxt('data/cars.csv', delimiter=',', unpack=True)
print(data)
print(data[1])