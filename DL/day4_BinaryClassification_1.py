import tensorflow as tf
tf.set_random_seed(777)

xdata=[[1,2],
       [2,3],
       [3, 1],
       [4, 3],
       [5, 3],
       [6, 2]]

ydata=[[0],# cast 사용결과 : 1=>false=>0
       [0],# 0=>true=>1
       [0],#0
       [1],#1
       [1],#1
       [1]]#1

x=tf.placeholder(tf.float32,shape=[None,2])
y=tf.placeholder(tf.float32,shape=[None,1])

w=tf.Variable(tf.random_normal([2,1]))
b=tf.Variable(tf.random_normal([1]))

hf=tf.sigmoid(tf.matmul(x,w)+b)
cost=-tf.reduce_mean(y*tf.log(hf)+(1-y)*tf.log(1-hf))
train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


# 0.5보다 크면 -> 1.0으로 캐스팅 그렇지 않으면 0으로 캐스팅

predicted=tf.cast(hf>0.5, dtype=tf.float32)
accracy=tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        costv, _ = sess.run([cost, train], feed_dict={x:xdata, y:ydata})
        if step % 200 == 0:
            print(step, costv)

    hfv, pv, av = sess.run([hf, predicted, accracy],feed_dict={x:xdata,y:ydata})
    print("\nhf: ", hfv, "\nPredicted: ", pv, "\nAccuracy: ", av)
