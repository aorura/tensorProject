import tensorflow as tf

def qRunner3():

    queue = tf.train.string_input_producer(['Data/trees.csv'], shuffle=False)
    reader=tf.TextLineReader(skip_header_lines=1)
    _, value=reader.read(queue=queue)

    record_defaults = [[-1], [99], [0]]
    x1, y, x2 = tf.decode_csv(value, record_defaults=record_defaults)

    x1batch, ybatch, x2batch = tf.train.batch([x1,y,x2], batch_size=5)

    sess = tf.Session()
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess, coord)

    x1 = tf.placeholder(tf.float32)
    x2 = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    w1 = tf.Variable(tf.random_normal([1]))
    w2 = tf.Variable(tf.random_normal([1]))
    b = tf.Variable(tf.random_normal([1]))

    hf = tf.multiply(x1, w1) + tf.multiply(x2,w2) + b

    cost = tf.reduce_mean(tf.square(hf - y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
    train = optimizer.minimize(cost)

    sess.run(tf.global_variables_initializer())

    for step in range(5):
        x1data, ydata, x2data = sess.run([x1batch, ybatch, x2batch])
        # cv, w1v, w2v, bv, hfv, _ = sess.run([cost, w1, w2, b, hf, train], feed_dict={x1: x1data, x2:x2data, y: ydata})
        # if step % 1000 == 0:
        #     print(step, "cost: ", cv, " w1: ", w1v, " w2: ", w2v, " b: ", bv)

    # xx1 = [8.3, 2.4, 9.9]
    # xx2 = [10.2, 23.4, 16.4]
    # predic = sess.run(hf, feed_dict={x1: xx1, x2:xx2})
    # print("\n(8.3,10.2)=> ", predic[0], " (2.4,23.4)=> ", predic[1], " (9.9,16.4)=> ", predic[2])

    sess.close()
    coord.request_stop()
    coord.join(threads=threads)

qRunner3()

