import tensorflow as tf
tf.set_random_seed(777)

filename_queue=tf.train.string_input_producer(['score.csv'], shuffle=False, name='filename_queue')
reader=tf.TextLineReader()
key, value=reader.read(filename_queue)
record_defaults=[[0.],[0.],[0.],[0.]]
xy=tf.decode_csv(value, record_defaults=record_defaults)

trainx, trainy=tf.train.batch([xy[0:-1], xy[-1:]],batch_size=10)

x=tf.placeholder(tf.float32, shape=[None, 3])
y=tf.placeholder(tf.float32, shape=[None, 1])

w=tf.Variable(tf.random_normal([3, 1]))
b=tf.Variable(tf.random_normal([1]))

hf=tf.matmul(x,w)+b
cost=tf.reduce_mean(tf.square(hf-y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)
sess=tf.Session()
sess.run(tf.global_variables_initializer())

#아래의 코드는 변함이 없음
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(101):
    xbatch, ybatch=sess.run([trainx, trainy])
    #print(step, "x 배치: ", xbatch)
    #print(step, "y 배치: ", ybatch)
    cv, hfv, _=sess.run([cost, hf, train], feed_dict={x:xbatch, y:ybatch})
    if step % 10 ==0:
        print(step, "cost:", cv, "\nPrediction:\n", hfv)

#아래의 코드는 변함이 없음
coord.request_stop()
coord.join(threads)

print("예측 점수 : ", sess.run(hf, feed_dict={x:[[90,80,85]]}))














