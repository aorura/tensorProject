import tensorflow as tf
import numpy as np

def read_iris():
    iris=np.loadtxt('iris_softmax.csv', delimiter=',')
    print(iris.shape)
    trainset=np.vstack((iris[:40], iris[50:90], iris[100:140]))
    testset = np.vstack((iris[40:50], iris[90:100], iris[140:]))
    print(trainset.shape)
    print(testset.shape)
    return trainset, testset

trainset, testset=read_iris()
#print(testset)
xx=trainset[:, :-3]
y=trainset[:, -3:]
x=tf.placeholder(tf.float32)

w=tf.Variable(tf.zeros([5,3]))
#      (120, 3) =(120, 5) * ( 5, 3)
z=tf.matmul(x,w)
hf=tf.nn.softmax(z)

cost=tf.reduce_mean(tf.reduce_sum(y*-tf.log(hf), axis=1))

optimizer=tf.train.GradientDescentOptimizer(0.1)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i  in range(2001):
    sess.run(train, feed_dict={x:xx})
    if i %20==0:
        print(i, sess.run(cost, feed_dict={x:xx}))
##################트레이닝(모델 생성) ###################

xx=testset[:, :-3] #(30,5)
y=testset[:, -3:] #(30,3)

yhat=sess.run(hf, feed_dict={x:xx})
print(yhat)

yhat2=sess.run(tf.argmax(yhat, axis=1))
print(yhat2) #예측

y2=sess.run(tf.argmax(y, axis=1))
print(y2) #정답

equal=sess.run(tf.equal(yhat2,y2))
print(equal)

cast=sess.run(tf.cast(equal, tf.float32))
print(cast)

mean=sess.run(tf.reduce_mean(cast))
print(mean)































