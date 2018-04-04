import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

xy=np.loadtxt("Data/score.csv", delimiter=',', dtype=np.float32)

xdata=xy[:,0:-1]
ydata=xy[:,[-1]]

# print(ydata)
# print(xdata)

print(xdata.shape, len(xdata))
print(ydata.shape, len(ydata))

x=tf.placeholder(tf.float32,shape=[None,3])
y=tf.placeholder(tf.float32, shape=[None,1])

w=tf.Variable(tf.random_normal([3,1]))
b=tf.Variable(tf.random_normal([1]))

hf=tf.matmul(x,w)+b

cost=tf.reduce_mean(tf.square(hf-y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    cv, hfv, _ = sess.run([cost, hf, train], feed_dict={x:xdata,y:ydata})
    # if step % 100 == 0:
        #print(step, "cost:", cv, "\nPrediction:\n", hfv)

xx=[[100,70,95],[60,70,90],[80,90,95]]
predic=sess.run(hf,feed_dict={x:xx})
print(predic)

sess.close()