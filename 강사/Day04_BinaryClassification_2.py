import numpy as np
import tensorflow as tf

def bc():
    #(6*3) * (3*1) = (6*1)
    xx=[[1., 1., 1., 1., 1., 1.],
        [2., 3., 3., 5., 7., 2.],
        [1., 2., 5., 5., 5., 5.]]
    yy=np.array([0,0,0,1,1,1])

    x=tf.placeholder(tf.float32)
    y=tf.placeholder(tf.float32)

    w=tf.Variable(tf.random_uniform([1,3], -1, 1))
    z=tf.matmul(w,x)

    hf=tf.nn.sigmoid(z)
    cost=-tf.reduce_mean(y*tf.log(hf)+
                         (1-y)*tf.log(1-hf))
    optimizer=tf.train.GradientDescentOptimizer(0.1)
    train=optimizer.minimize(cost)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train, feed_dict={x:xx, y:yy})
        if i % 100 == 0:
            print(i, sess.run(cost, feed_dict={x:xx,y:yy}))

    xx=[[1., 1.],
        [3., 7.],
        [8., 2.]]
    print(sess.run(hf, feed_dict={x:xx}))
    y_hat=sess.run(hf, feed_dict={x:xx})
    print(y_hat>0.5)
    print(sess.run(w))
bc()





# import math
# def sigmoid(z):
#     return 1/(1+math.e**-z)
# ww=sess.run(w)
# z=np.dot(ww, xx)
# print(z)
# print(sigmoid(z))
# sess.close()