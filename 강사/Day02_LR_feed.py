import tensorflow as tf
tf.set_random_seed(777)

w=tf.Variable(tf.random_normal([1]))
b=tf.Variable(tf.random_normal([1]))

x=tf.placeholder(tf.float32, shape=[None])
y=tf.placeholder(tf.float32, shape=[None])

hf=x*w+b
cost=tf.reduce_mean(tf.square(hf-y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    costv, wv, bv, _ =sess.run([cost, w, b, train],
                               feed_dict={x:[1,2,3,4,5], y:[2.1,3.3,4.2,5.4,6.2]})
    if step % 100 ==0:
        print(step, costv, wv, bv)

print(sess.run(hf, feed_dict={x:[5]}))  #x 값이 10일때 예측되는 y값을 출력하세요
print(sess.run(hf, feed_dict={x:[2.5]}))  # x 값이 2.5일때 예측되는 y값을 출력하세요
print(sess.run(hf, feed_dict={x:[1.5, 3.8]}))

# for step in range(2001):
#     costv, wv, bv, _ =sess.run([cost, w, b, train],
#                                feed_dict={x:[1,2,3], y:[1,2,3]})
#     if step % 100 ==0:
#         print(step, costv, wv, bv)
#
# print(sess.run(hf, feed_dict={x:[10]}))  #x 값이 10일때 예측되는 y값을 출력하세요
# print(sess.run(hf, feed_dict={x:[2.5]}))  # x 값이 2.5일때 예측되는 y값을 출력하세요























