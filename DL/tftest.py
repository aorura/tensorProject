import tensorflow as tf

hello=tf.constant('hi')
sess=tf.Session()
print(sess.run(hello))