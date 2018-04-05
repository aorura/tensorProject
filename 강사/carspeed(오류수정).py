import tensorflow as tf
import numpy as np
import csv
tf.set_random_seed(777)

filenames = 'Data/cars.csv'




def makeBatch(filenames, batch_size=5, isshuffle=False):
    q = tf.train.string_input_producer([filenames], shuffle=isshuffle)
    reader = tf.TextLineReader()
    _, value = reader.read(queue=q)
    record_defaults = [[0], [0]]


    xy = tf.decode_csv(value, record_defaults=record_defaults)

    xbatch, ybatch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=batch_size)



    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    x, y = sess.run([xbatch, ybatch])
    print(x, y)
    # for i in range(101):
    #     x, y = sess.run([xbatch, ybatch])
    #     print(x, y)

    coord.request_stop()
    coord.join(threads=threads)

    # return xbatch, ybatch





makeBatch(filenames, batch_size = 5)





# w = tf.Variable(tf.random_normal([1]), name="weight")
# b = tf.Variable(tf.random_normal([1]), name="bias")
#
# x = tf.placeholder(dtype=tf.float32,  shape=[None])
# y = tf.placeholder(dtype=tf.float32,  shape=[None])
#
# hf = tf.matmul(x, w) + b
#
#
# cost = tf.reduce_mean(tf.square(hf-y))
# optimizer  = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
# train = optimizer.minimize(cost)