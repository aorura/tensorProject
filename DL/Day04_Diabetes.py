import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(777)

xy=np.loadtxt("data2/diabetes.csv", delimiter=',',dtype=np.float32)

xy_seventy=int(len(xy)*0.7)
xy_end=len(xy)

xdata=xy[:xy_seventy,0:-1]
ydata=xy[:xy_seventy,[-1]]

xtest=xy[xy_seventy:,0:-1]
ytest=xy[xy_seventy:,[-1]]

print(xdata.shape,ydata.shape)

x=tf.placeholder(tf.float32, shape=[None,8])
y=tf.placeholder(tf.float32, shape=[None,1])

w=tf.Variable(tf.random_normal([8,1]))
b=tf.Variable(tf.random_normal([1]))

z=tf.matmul(x,w)+b
hf=tf.sigmoid(z)
cost=-tf.reduce_mean(y*tf.log(hf)+(1-y)*tf.log(1-hf))
train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted=tf.cast(hf>0.5, dtype=tf.float32)
accracy=tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        costv, _= sess.run([cost, train], feed_dict={x:xdata,y:ydata})
        if step % 200 == 0:
            print(step, costv)

    hfv, pv, av = sess.run([hf, predicted, accracy], feed_dict={x: xtest, y: ytest})
    print("\nhf: ", hfv, "\nPredicted: ", pv, "\nAccuracy: ", av)

