#털과 날개가 있는지 없는지에 따라, 포유류/조류인지 분류하는
#딥러닝 모델 설계
#신경망의 레이어를 여러개로 구성
import tensorflow as tf
import numpy as np

#[털, 날개]
xdata=np.array(
    [[0,0], [1,0], [1,1], [0,0], [0,0], [0,1]])
#[기타, 포유류, 조류]
ydata=np.array([
    [1,0,0], #기타
    [0,1,0], #포유류
    [0,0,1], #조류
    [1,0,0],
    [1,0,0],
    [0,0,1]
])
#모델 구성

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
#첫번째 가중치의 차원은 [특성, 히든 레이어의 뉴런갯수]->[2,10]으로 한다
w1=tf.Variable(tf.random_uniform([2,10], -1., 1.))
#두번째 가중치의 차원은 [첫번째 히든 레이어의 뉴런갯수, 분류 개수]->[10,3]으로 한다
w2=tf.Variable(tf.random_uniform([10,3], -1., 1.))

#편향을 각 레이어의 출력 개수로 설정
b1=tf.Variable(tf.zeros([10]))
b2=tf.Variable(tf.zeros([3]))

#신경망의 히든 레이어에 가중치 w1 과 b1을 적용

L1=tf.add(tf.matmul(x, w1),b1)
L1=tf.nn.relu(L1)

model=tf.add(tf.matmul(L1, w2), b2)

cost=tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)

#######신경망 모델 학습

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(101):
    sess.run(train, feed_dict={x:xdata,y:ydata})

    if step % 10==0:
        print(step+1, sess.run(cost,feed_dict={x:xdata,y:ydata}))

prediction=tf.argmax(model,1)
target=tf.argmax(y,1)
print("예측값:",sess.run(prediction, feed_dict={x:xdata}))
print("실제값:",sess.run(target, feed_dict={y:ydata}))

iscorrect=tf.equal(prediction, target)
accuracy=tf.reduce_mean(tf.cast(iscorrect,tf.float32))
print("정확도: %.2f" % sess.run(accuracy*100, feed_dict={x:xdata,y:ydata}))












































































