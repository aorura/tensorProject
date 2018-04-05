import tensorflow as tf
tf.set_random_seed(777)
ydata=[4,5,6]
xdata=[2,3,4]
w=tf.Variable(tf.random_normal([1]), name="weight")

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

hf=x*w
cost=tf.reduce_mean(tf.square(hf-y))
learning_rate=0.01

gradient=tf.reduce_mean((w*x-y)*x)
descent=w-learning_rate*gradient
update=w.assign(descent)
# 같은 의미
# optimizer=tf.train.GradientDescentOptimizer(0.01)
# train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(201):
    sess.run(update, feed_dict={x:xdata, y:ydata})
    print(step, sess.run(cost, feed_dict={x:xdata, y:ydata}), sess.run(w))
















