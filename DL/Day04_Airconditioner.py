import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def normal_scaler(data):
    # print(np.min(data), np.max(data))
    # print(np.min(data,axis=0)) # axis=0 : 열 중에서 가장 작은값, axis=1: 행에서 가장 작은 값
    min=np.min(data, axis=0)
    max = np.max(data, axis=0)

    return ((data-min)/(max-min))

tf.set_random_seed(777)

xy=np.loadtxt("data2/aircon.csv", delimiter=',',dtype=np.float32, skiprows=1)

data=normal_scaler(xy)

data = np.transpose(data)
xdata = data[:-1].transpose().astype(np.float32)
ydata = data[-1:].transpose().astype(np.float32)

w = tf.Variable(tf.random_uniform([2, 1], -1, 1))
b = tf.Variable(tf.random_uniform([1], -1, 1))

# xdata=xy[:,0:-1]
# ydata=xy[:,[-1]]

x=tf.placeholder(tf.float32, shape=[None,2])
y=tf.placeholder(tf.float32, shape=[None,1])

z=tf.matmul(x,w)+b
hf=tf.sigmoid(z)
cost=-tf.reduce_mean(y*tf.log(hf)+(1-y)*tf.log(1-hf))
train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted=tf.cast(hf>0.5, dtype=tf.float32)
accracy=tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        costv, wv, bv, _= sess.run([cost, w, b, train], feed_dict={x:xdata,y:ydata})
        if step % 200 == 0:
            print("Setp: ", step, " cost: ", costv, " w: ", wv, " b: ", bv)

    hfv, pv, av = sess.run([hf, predicted, accracy], feed_dict={x: xdata, y: ydata})
    print("\nhf: ", hfv, "\nPredicted: ", pv, "\nAccuracy: ", av)

    testData=[[27,50], [31,20],[22,70]]
    result=sess.run(hf, feed_dict={x:testData})

    print("27도, 50%: ", result[0], "31도, 20%: ", result[1], "22도, 70%: ", result[2])

