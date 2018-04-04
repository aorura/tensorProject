import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

data=np.loadtxt('Data/cars.csv', delimiter=',', unpack=True, skiprows=1)

xdata=data[0]
ydata=data[1]

print(ydata)
print(xdata)

print(xdata.shape, len(xdata))
print(ydata.shape, len(ydata))

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

w=tf.Variable(tf.random_normal([1]))
b=tf.Variable(tf.random_normal([1]))

hf=tf.multiply(x,w)+b

cost=tf.reduce_mean(tf.square(hf-y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-4)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    cv, wv, bv, hfv, _ = sess.run([cost, w, b, hf, train], feed_dict={x:xdata,y:ydata})
    if step % 100 == 0:
        print(step, "cost: ", cv, " w: ", wv, " b: ", bv)

xx=[25,30,2]
predic=sess.run(hf,feed_dict={x:xx})
print("\nspeed 25=> ", predic[0], " speed 30=> ", predic[1], " speed 2=> ", predic[2])

sess.close()