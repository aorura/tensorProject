import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

data=np.loadtxt('Data/trees.csv', delimiter=',', unpack=True, skiprows=1)

x1data=data[0]
x2data=data[2]
ydata=data[1]

print(ydata)
print(x1data)
print(x2data)

print(x1data.shape, len(x1data))
print(ydata.shape, len(ydata))

x1=tf.placeholder(tf.float32)
x2=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

w1=tf.Variable(tf.random_normal([1]))
w2=tf.Variable(tf.random_normal([1]))
b=tf.Variable(tf.random_normal([1]))

hf=tf.multiply(x1,w1) + tf.multiply(x2,w2) +b

cost=tf.reduce_mean(tf.square(hf-y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-4)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    cv, hfv, _ = sess.run([cost, hf, train], feed_dict={x1:x1data, x2:x2data,y:ydata})
    if step % 1000 == 0:
        print(step, "cost: ", cv, "\nhfv: ", hfv)

# xx=[25,30,2]
# predic=sess.run(hf,feed_dict={x:xx})
# print("\nspeed 25=> ", predic[0], " speed 30=> ", predic[1], " speed 2=> ", predic[2])

sess.close()