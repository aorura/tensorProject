import tensorflow as tf

# hello=tf.constant('hi')
sess=tf.Session()
# print(sess.run(hello))
# print(str(sess.run(hello), encoding='utf-8'))

# a=tf.constant(5)
# b=tf.constant(3)
# c=tf.multiply(a,b)
# d=tf.add(a,b)
# e=tf.add(c,d)
# print(sess.run(e))

# a=tf.constant([5])
# b=tf.constant([3])
# c=tf.constant([2])
# d=a*b+c
# print(sess.run(d))

inputdata=[1,2,3,4,5]
x=tf.placeholder(dtype=tf.float32)
w=tf.Variable([2], dtype=tf.float32)
sess.run(tf.global_variables_initializer())
y=w*x
res=sess.run(y,feed_dict={x:inputdata})
print(res)
