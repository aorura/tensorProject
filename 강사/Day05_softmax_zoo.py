import tensorflow as tf
import numpy as np
tf.set_random_seed(777)
xy=np.loadtxt('zoo.csv', delimiter=',', dtype=np.float32)
xdata=xy[:, 0:-1]
ydata=xy[:, [-1]]
print(xdata.shape, ydata.shape)

nb_classes=7 #동물 종류 0~6

x=tf.placeholder(tf.float32, [None,16])
y=tf.placeholder(tf.int32, [None, 1]) #0-> 1000000
yonehot=tf.one_hot(y, nb_classes)
#원핫 인코딩을 수행하면 한 차원 높게 변환됨. 즉,(None,1,7)
#예) [[1],[3]] ->  None,1인 shape을 원핫 인코딩 수행 후,
#[[[0100000], [[0001000]]]이 됨.rank : 2 -> rank : 3
#[None,7]가 우리가 원하는 형태임. reshape함수 사용하면 됨.
#예) tf.reshape(yonehot,[-1, 7])로 주면 shape은(None,7)이 됨
#최종결과 : [[0100000], [[0001000]]
print("one hot : ", yonehot)
yonehot=tf.reshape(yonehot, [-1, nb_classes])
print("one hot reshape : ", yonehot)

w=tf.Variable(tf.random_normal([16, nb_classes]))
b=tf.Variable(tf.random_normal([nb_classes]))

logits=tf.matmul(x,w)+b
hf=tf.nn.softmax(logits)

costi=tf.nn.softmax_cross_entropy_with_logits\
    (logits=logits, labels=yonehot)
cost=tf.reduce_mean(costi)
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction=tf.argmax(hf,1)
correct_prediction=tf.equal(prediction, tf.argmax(yonehot, 1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(optimizer, feed_dict={x:xdata, y:ydata})
        if step %100==0:
            cv, av = sess.run([cost, accuracy], feed_dict={
                x: xdata, y: ydata})
            print("step: {:5}\tcost:{:.3f}\tacc:{:.2%}".format(step,cv,av))
    pred=sess.run(prediction, feed_dict={x:xdata})
    for p,y in zip(pred, ydata.flatten()):
        print("[{}] prediction:{} True y: {}". format(p==int(y),p,int(y)))
        #ydata:(None,1) -> flatten(None,)
        #[[1],[0]] -> [1,0]으로 평평하게 해줌










































