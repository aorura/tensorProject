# import tensorflow as tf
#
# w=tf.Variable([.3], tf.float32)
# b=tf.Variable([-.3], tf.float32)
#
# x=tf.placeholder(tf.float32)
# y=tf.placeholder(tf.float32)
#
# hf=x*w+b
#
# loss=tf.reduce_mean(tf.square(hf-y))
#
# optimizer=tf.train.GradientDescentOptimizer(0.01)
#
# train=optimizer.minimize(loss)
#
# xtrain=[1,2,3,4]
# ytrain=[0,-1,-2,-3]
#
# init=tf.global_variables_initializer()
# sess=tf.Session()
# sess.run(init)
#
# for i in range(2001):
#     sess.run(train, {x:xtrain, y:ytrain})
#
# wv, bv, lossv = sess.run([w,b,loss], feed_dict={x:xtrain, y:ytrain})
# print("weight: %s bias: %s loss: %s" % (wv, bv, lossv))

import tensorflow as tf

tf.set_random_seed(777)
xdata=[4,5,6]
ydata=[2,3,4]
w=tf.Variable(tf.random_normal([1]), name="weight")

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

hf=x*w
cost=tf.reduce_mean(tf.square(hf-y))
learning_rate=0.01


gradient=tf.reduce_mean((w*x-y)*x)
descent=w-learning_rate*gradient
update=w.assign(descent)   # assign => 변수에 값 할당
# 위 코드와 같은 의미
# optimizer=tf.train.GradientDescentOptimizer(0.01)
# train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(201):
    sess.run(update, feed_dict={x:xdata,y:ydata})
    print(step, sess.run(cost, feed_dict={x:xdata,y:ydata}), sess.run(w))