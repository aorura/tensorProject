import tensorflow as tf

tf.set_random_seed(777)

xdata=[[70.,80.,75.],[90.,92.,87.],[83.,88.,92.], [95.,94.,100.],[72.,66.,77.]] # 5*3


ydata=[[280.],[360.],[348.],[365.],[240.]] # 5*1

x=tf.placeholder(tf.float32, shape=[None, 3])
y=tf.placeholder(tf.float32, shape=[None, 1])

w=tf.Variable(tf.random_normal([3,1]), name="weight")

b=tf.Variable(tf.random_normal([1]), name="bias")

hf=tf.matmul(x, w)+b
cost=tf.reduce_mean(tf.square(hf-y))

optimizer =tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

sess=tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    cv, hfv, _= sess.run([cost,hf,train],feed_dict={x:xdata,y:ydata})
    if step % 10 == 0:
        print(step, "cost:", cv, "\nPrediction:\n", hfv)