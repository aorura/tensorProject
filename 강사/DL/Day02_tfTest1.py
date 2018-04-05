# import tensorflow as tf
# hello=tf.constant('hi')
# sess=tf.Session()
# print(sess.run(hello))
# print(str(sess.run(hello), encoding='utf-8'))
# #
# # # a=tf.constant(5)
# # # b=tf.constant(3)
# # # c=tf.multiply(a,b)
# # # d=tf.add(a,b)
# # # e=tf.add(c,d)
# # # print(sess.run(e))
# #
# # a=tf.constant([5])
# # b=tf.constant([3])
# # c=tf.constant([2])
# # d=a*b+c
# # print(sess.run(d))
#
# inputdata=[1,2,3,4,5]
# x=tf.placeholder(dtype=tf.float32)
# w=tf.Variable([2], dtype=tf.float32)
# y=w*x
# sess.run(tf.global_variables_initializer())
# res=sess.run(y, feed_dict={x:inputdata})
# print(res)
#
#

import matplotlib.pyplot as plt

def costf(x,y,w):
    cost=0
    for i in range(len(x)):
        hx=w*x[i]
        cost+=(hx-y[i])**2
    return cost/len(x)

x=[1,2,3]
y=[1,2,3]
w=10
print(costf(x,y,w))
print(costf(x,y,5))
xx, yy = [],[]
for i in range(-50, 50):
    w=i/10
    costv=costf(x,y,w)
    print(w, costv)
    xx.append(w)
    yy.append(costv)
plt.plot(xx,yy)
plt.plot(xx,yy, 'ro')
plt.show()











