import tensorflow as tf

def qRunner3():

    queue = tf.train.string_input_producer(['Data/cars.csv'], shuffle=False)
    reader=tf.TextLineReader(skip_header_lines=1)
    _, value=reader.read(queue=queue)

    record_defaults = [[-1], [99]]
    x, y = tf.decode_csv(value, record_defaults=record_defaults)

    xbatch, ybatch = tf.train.batch([x,y], batch_size=5)

    sess = tf.Session()
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess, coord)

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    w = tf.Variable(tf.random_normal([1]))
    b = tf.Variable(tf.random_normal([1]))

    hf = tf.multiply(x, w) + b

    cost = tf.reduce_mean(tf.square(hf - y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
    train = optimizer.minimize(cost)

    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        xdata, ydata = sess.run([xbatch, ybatch])
        cv, wv, bv, hfv, _ = sess.run([cost, w, b, hf, train], feed_dict={x: xdata, y: ydata})
        if step % 1000 == 0:
            print(step, "cost: ", cv, " w: ", wv, " b: ", bv)

    xx = [25, 30, 2]
    predic = sess.run(hf, feed_dict={x: xx})
    print("\nspeed 25=> ", predic[0], " speed 30=> ", predic[1], " speed 2=> ", predic[2])

    sess.close()
    coord.request_stop()
    coord.join(threads=threads)

qRunner3()

