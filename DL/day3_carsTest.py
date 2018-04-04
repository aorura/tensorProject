import tensorflow as tf

# def qRunner3():
#
#     queue = tf.train.string_input_producer('cars.csv', shuffle=False)
#     reader=tf.TextLineReader()
#     _, value=reader.read(queue=queue)
#
#     record_defaults = [[-], [99]]
#     x, y = tf.decode_csv(value, record_defaults=record_defaults)
#
#     xbatch, ybatch = tf.train.batch([x,y], batch_size=5)
#
#     sess = tf.Session()
#     coord=tf.train.Coordinator()
#     threads=tf.train.start_queue_runners(sess, coord)
#
#     for i in range(20):
#         x,y=sess.run([xbatch,ybatch])
#         print(x,y)
#
#     coord.request_stop()
#     coord.join(threads=threads)
#
# qRunner3()

