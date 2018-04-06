import tensorflow as tf
import random
import matplotlib.pyplot as plt
#epoch(에폭) : 전체 훈련 데이터셋을 1회 트레이닝 했을 경우->1에폭
#batch_size(트레이닝 단위):100개씩 트레이닝
tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate=0.001
training_epochs=15
batch_size=100

x=tf.placeholder(tf.float32, [None, 784])#28*28
y=tf.placeholder(tf.float32, [None, 10])#0~9

w=tf.Variable(tf.random_normal([784, 10]))
b=tf.Variable(tf.random_normal([10]))

hf=tf.matmul(x,w)+b

cost=tf.reduce_mean\
    (tf.nn.softmax_cross_entropy_with_logits(
        logits=hf, labels=y
    ))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost=0
    total_batch=int(mnist.train.num_examples/batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys=mnist.train.next_batch(batch_size)
        feed_dict={x:batch_xs, y:batch_ys}
        cv, _=sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost+=cv/total_batch
    print('Epoch:', '%04d' % (epoch+1), 'cost:', '{:.9f}'.format(avg_cost))
print('학습 종료됨')

correct_prediction=tf.equal(tf.argmax(hf,1), tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('accuracy:', sess.run(accuracy, feed_dict={
    x:mnist.test.images, y:mnist.test.labels}))

r=random.randint(0, mnist.test.num_examples-1)
print("Label:",sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
print("Prediction:", sess.run(tf.argmax(hf, 1), feed_dict={x:mnist.test.images[r:r+1]}))

plt.imshow(mnist.test.images[r:r+1].reshape(28,28),
           cmap='Greys', interpolation='nearest')
plt.show()






































