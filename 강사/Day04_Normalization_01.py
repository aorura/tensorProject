import tensorflow as tf
import numpy as np

def no_normal():
    data =     [[828.659973, 833.450012, 908100, 828.349976, 831.659973],
                [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
                [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
                [816, 820.958984, 1008100, 815.48999, 819.23999],
                [819.359985, 823, 1188100, 818.469971, 818.97998],
                [819, 823, 1198100, 816, 820.450012],
                [811.700012, 815.25, 1098100, 809.780029, 813.669983],
                [809.51001, 816.659973, 1398100, 804.539978, 809.559998]]
    data=np.transpose(data)#8,5 -> 5,8
    print(data.shape)
    print("===========================")
    print(data[:-1])
    x=data[:-1].transpose().astype(np.float32)
    y=data[-1:].transpose().astype(np.float32)
    print("===========================")
    print(x)
    print("===========================")
    print(y)
    print("===========================")
    print(x.shape)
    print(y.shape)

    w=tf.Variable(tf.random_uniform([4, 1], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))
    #        (8,1) = (8,4) * (4, 1)
    hf=tf.matmul(x,w)+b
    cost=tf.reduce_mean((hf-y)**2)
    learaning_rate=0.0001
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learaning_rate)
    train=optimizer.minimize(cost)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(201):
        sess.run(train)
        if i % 10 ==0:
            print(i, sess.run(cost))

    sess.close()

def normal_scaler(data):   #normalization
    #print(np.min(data))
    #print(np.min(data, axis=0))
    #print(np.max(data))
    min=np.min(data, axis=0)
    max=np.max(data,axis=0)
    print(min)
    print("--------")
    print(max)

    return (data-min)/(max-min)

data =     [[828.659973, 833.450012, 908100, 828.349976, 831.659973],
            [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
            [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
            [816, 820.958984, 1008100, 815.48999, 819.23999],
            [819.359985, 823, 1188100, 818.469971, 818.97998],
            [819, 823, 1198100, 816, 820.450012],
            [811.700012, 815.25, 1098100, 809.780029, 813.669983],
            [809.51001, 816.659973, 1398100, 804.539978, 809.559998]]
print(normal_scaler(data))
#no_normal()
data=normal_scaler(data)
print(data.shape)

data = np.transpose(data)  # 8,5 -> 5,8
print(data.shape)
print("===========================")
print(data[:-1])
x = data[:-1].transpose().astype(np.float32)
y = data[-1:].transpose().astype(np.float32)
print("===========================")
print(x)
print("===========================")
print(y)
print("===========================")
print(x.shape)
print(y.shape)

w = tf.Variable(tf.random_uniform([4, 1], -1, 1))
b = tf.Variable(tf.random_uniform([1], -1, 1))
#        (8,1) = (8,4) * (4, 1)
hf = tf.matmul(x, w) + b
cost = tf.reduce_mean((hf - y) ** 2)
learaning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learaning_rate)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(201):
    sess.run(train)
    if i % 10 == 0:
        print(i, sess.run(cost))

sess.close()

