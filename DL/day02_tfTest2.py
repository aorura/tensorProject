# import tensorflow as tf
#
# a=tf.placeholder("float")
# b=tf.placeholder("float")
# y=tf.multiply(a,b)
# z=tf.add(y,y)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(y, feed_dict={a:2, b:1}))
#     print(sess.run(z, feed_dict={a:3, b:4}))
#
#


# import tensorflow as tf
# a=tf.placeholder("float")
# b=tf.placeholder("float")
# y=tf.multiply(a,b)
# z=tf.add(y,y)
#
# with tf.Session() as sess:
#     print(sess.run([y,z], feed_dict={a:3,b:2}))

import tensorflow as tf

x=tf.constant(30)
y=tf.Variable(x+5)

x2=tf.constant([[1.,2.,3.]]) # 1행 3열
w= tf.constant([[2.], [2.], [2.]])  # 3행 1열

y2=tf.matmul(x2,w)
print(x2.get_shape())
sess=tf.Session()
sess.run(tf.global_variables_initializer())
res=sess.run(y2)
print(res)

inputdata=[[1.,2.,3.],[3.,4.,2.],[5.,2.,4.]]
x3=tf.placeholder(dtype=tf.float32, shape=[None, 3])
w3=tf.Variable([[2.],[2.],[2.]], dtype=tf.float32)
y3=tf.matmul(x3,w3)
sess.run(tf.global_variables_initializer())
res=sess.run(y3, feed_dict={x3:inputdata})
print(res)