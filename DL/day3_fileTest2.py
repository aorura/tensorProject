import numpy as np

data=np.loadtxt('Data/cars.csv', delimiter=',', unpack=False, skiprows=1)    # unpack = True열단위로 읽음, skiprows = 건너뛰기
print(data)