import tensorflow as tf
tf.set_random_seed(777)
x1data=[70., 90., 83., 95., 72. ]#1회 모의고사
x2data=[80., 92., 88., 94., 66. ]#2회 모의고사
x3data=[75., 87., 92., 100., 77.]#3회 모의고사
ydata=[280., 360., 348., 365., 240.]#수능점수

x1=tf.placeholder(tf.float32)
x2=tf.placeholder(tf.float32)
x3=tf.placeholder(tf.float32)

y=tf.placeholder(tf.float32)

w1=tf.Variable(tf.random_normal([1]), name='weight1')
w2=tf.Variable(tf.random_normal([1]), name='weight2')
w3=tf.Variable(tf.random_normal([1]), name='weight3')

b=tf.Variable(tf.random_normal([1]), name='bias')

hf=x1*w1+x2*w2+x3*w3+b
cost=tf.reduce_mean(tf.square(hf-y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cv, hfv, _=sess.run([cost, hf, train],
             feed_dict={x1:x1data, x2:x2data, x3:x3data, y:ydata})
    if step % 10 ==0:
        print(step, "cost:", cv, "\nPrediction:\n", hfv)



















