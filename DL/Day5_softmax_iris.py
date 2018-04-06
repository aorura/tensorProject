import tensorflow as tf
import numpy as np

def read_iris():
    iris=np.loadtxt('data2/iris_softmax.csv', delimiter=',')
    print(iris.shape)
    trainset=np.vstack((iris[:40], iris[50:90],iris[100:140])) # vstack: 행단위 띄엄띄엄 데이터 추출
    testset = np.vstack((iris[40:50], iris[90:100], iris[140:]))
    print(trainset.shape)
    print(testset.shape)
    return trainset,testset

trainset, testset = read_iris()
# print(testset)

xx=trainset[:,:-3]
y=trainset[:,-3:]
x=tf.placeholder(tf.float32)

w=tf.Variable(tf.zeros([5,3]))

# (120,5)*(5,3)=(120,3)
z=tf.matmul(x,w)
hf=tf.nn.softmax(z)
cost=tf.reduce_mean(tf.reduce_sum(y*-tf.log(hf),axis=1))

train=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2001):
        sess.run(train, feed_dict={x:xx})
        if i % 20 == 0:
            print(i, sess.run(cost, feed_dict={x:xx}))
    print("::::::::::::: test ::::::::::::")
    xx=testset[:,:-3]
    y=testset[:,-3:]

    yhat=sess.run(hf, feed_dict={x:xx})
    print(yhat)
    print(":::::::::::: yhat2 ::::::::::::::")
    yhat2=sess.run(tf.argmax(yhat, axis=1)) # argmax  열 중 가장 큰 수 인덱스 리턴
    print(yhat2)
    print(":::::::::: y2 ::::::::::::::")
    y2=sess.run(tf.argmax(y, axis=1))
    print(y2)
    print(":::::::::::: equal ::::::::::::::")
    equal = sess.run(tf.equal(yhat2,y2))
    print(equal)

    cast=sess.run(tf.cast(equal, tf.float32))
    print(cast)
    print(":::::::::::: mean :::::::::::::::::")
    mean = sess.run(tf.reduce_mean(cast))
    print(mean)