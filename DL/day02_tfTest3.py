import tensorflow as tf

tf.set_random_seed(777)

xtrain=[1,2,3]
ytrain=[1,2,3]

#가설 : h(x) = wx+b

w=tf.Variable(tf.random_normal([1]))
b=tf.Variable(tf.random_normal([1]))

hf= xtrain*w+b

cost=tf.reduce_mean(tf.square(hf-ytrain))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost) #  optimizer를 이용하여 cost에 대해 찾아라

############ 여기까지가 그래프 빌드 ##############

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)

    if step % 100 == 0:
        print(step, sess.run(cost), sess.run(w), sess.run(b))

