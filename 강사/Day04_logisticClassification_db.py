import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy=np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
xdata=xy[:,0:-1]
ydata=xy[:, [-1]]

print(xdata.shape, ydata.shape)

x=tf.placeholder(tf.float32, shape=[None, 8])
y=tf.placeholder(tf.float32, shape=[None, 1])

w=tf.Variable(tf.random_normal([8,1]))
b=tf.Variable(tf.random_normal([1]))

hf=tf.sigmoid(tf.matmul(x,w)+b)#예측값 범위: 0 ~ 1
cost=-tf.reduce_mean(y*tf.log(hf)+(1-y)*tf.log(1-hf))

train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize((cost))

predicted=tf.cast(hf>0.5, dtype=tf.float32)
accuracy=tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        costv, _=sess.run([cost, train], feed_dict={x:xdata, y:ydata})
        if step % 200 ==0:
            print(step, costv)
    hfv, pv, av=sess.run([hf, predicted, accuracy],
                         feed_dict={x:xdata, y:ydata})
    print("\nHypothesis:", hfv, "\nPrediction:", pv, "\nAccuracy:", av)


"""
0.5 초과 -> 1, 0.5 이하 -> 0
[1.]
[1.]
[0.]
...
[1.]
Accuracy : 0.xxxxx

"""




















