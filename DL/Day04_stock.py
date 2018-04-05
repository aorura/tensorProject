import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# def normal_scaler(data):
#     # print(np.min(data), np.max(data))
#     # print(np.min(data,axis=0)) # axis=0 : 열 중에서 가장 작은값, axis=1: 행에서 가장 작은 값
#     min=np.min(data, axis=0)
#     max = np.max(data, axis=0)
#
#     return ((data-min)/(max-min))

tf.set_random_seed(777)

xy=np.loadtxt("data2/stock_daily.csv", delimiter=',',dtype=np.float32, skiprows=2)

xdata = xy[:,0:-1]
ydata = xy[:,[-1]]

w = tf.Variable(tf.random_normal([4, 1]))
b = tf.Variable(tf.random_normal([1]))

x=tf.placeholder(tf.float32, shape=[None,4])
y=tf.placeholder(tf.float32, shape=[None,1])

hf=tf.matmul(x,w)+b
cost=tf.reduce_mean(tf.square(hf-y))
train=tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        costv, wv, bv, _= sess.run([cost, w, b, train], feed_dict={x:xdata,y:ydata})
        if step % 200 == 0:
            print("Setp: ", step, " cost: ", costv, " w: ", wv, " b: ", bv)



