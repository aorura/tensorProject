import tensorflow as tf

w=tf.Variable([.3], tf.float32)
b=tf.Variable([-.3], tf.float32)

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

hf=x*w+b

loss=tf.reduce_mean(tf.square(hf-y))
optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)

xtrain=[1,2,3,4]
ytrain=[0,-1,-2,-3]

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(2001):
    sess.run(train, {x:xtrain, y:ytrain})

wv, bv, lossv=sess.run([w, b, loss], feed_dict={x:xtrain,y:ytrain})
print("weight: %s bias: %s loss: %s" % (wv, bv, lossv))






































