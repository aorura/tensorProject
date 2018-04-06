import numpy as np
import matplotlib.pyplot as plt

#활성화함수 - 계단함수
def step_function(x):
    return np.array(x>0, dtype=np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

x=np.arange(-5.0, 5.0, 0.1)
#y=step_function(x)
#y=sigmoid(x)
y=relu(x)
plt.plot(x,y)
plt.ylim(-1.0,5.5)
plt.show()


#
# def XOR(x1,x2):
#     s1=NAND(x1,x2)
#     s2=OR(x1,x2)
#     y=AND(s1,s2)
#     return y
#
# def NAND(x1,x2):
#     x=np.array([x1,x2])
#     w=np.array([-0.5, -0.5])
#     b=0.7
#     tmp=np.sum(w*x)+b
#     if tmp<=0:
#         return 0
#     else:
#         return 1
#
# def AND(x1,x2):
#     x=np.array([x1,x2])
#     w=np.array([0.5, 0.5])
#     b=-0.7
#     tmp=np.sum(w*x)+b
#     if tmp<=0:
#         return 0
#     else:
#         return 1
#
# def OR(x1,x2):
#     x=np.array([x1,x2])
#     w=np.array([0.5, 0.5])
#     b=-0.2
#     tmp=np.sum(w*x)+b
#     if tmp<=0:
#         return 0
#     else:
#         return 1
#
# if __name__=='__main__':
#     for xs in [(0,0),(0,1),(1,0),(1,1)]:
#         y=XOR(xs[0], xs[1])
#         print(str(xs)+ "->", str(y))
