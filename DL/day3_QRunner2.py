import tensorflow as tf

tf.set_random_seed(777)

def qRunner1():
    sess=tf.Session()
    q=tf.train.string_input_producer(['10','20','30'], shuffle=False)
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess, coord)
    for i in range(11):
        value=sess.run(q.dequeue())
        print(value.decode('utf-8'))
    coord.request_stop()
    coord.join(threads=threads)

# qRunner1()

def qRunner2():
    sess=tf.Session()
    q = tf.train.string_input_producer(['Data/q_1.txt', 'Data/q_2.txt', 'Data/q_3.txt'], shuffle=False)
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess, coord)
    reader=tf.TextLineReader()
    key, value=reader.read(queue=q)

    # print(key)
    # print(value)
    # print(sess.run(key))
    # print(sess.run(value))

    record_defaults=[[0],[0]]
    for i in range(101):
        x, y=tf.decode_csv(value, record_defaults=record_defaults)
        print(sess.run([x,y]))
    coord.request_stop()
    coord.join(threads=threads)

# qRunner2()

def qRunner3():

    queue = tf.train.string_input_producer(['Data/q_1.txt', 'Data/q_2.txt', 'Data/q_3.txt'], shuffle=False)
    reader=tf.TextLineReader()
    _, value=reader.read(queue=queue)

    record_defaults = [[-1], [99]]
    x, y = tf.decode_csv(value, record_defaults=record_defaults)

    xbatch, ybatch = tf.train.batch([x,y], batch_size=5)

    sess = tf.Session()
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess, coord)

    for i in range(20):
        x,y=sess.run([xbatch,ybatch])
        print(x,y)


    coord.request_stop()
    coord.join(threads=threads)

qRunner3()