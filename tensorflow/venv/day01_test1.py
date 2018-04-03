# # 1
# # 1 2
# # 1 2 3
# # 1 2 3 4
# for i in range(1,5):
#     for j in range(1, i+1):
#         print(j, end=' ');
#     print();
#
#
# def order(a, b):
#     if a<b:
#         return a, b
#     else:
#         return b, a
#
# # not interest, use '_'
# _, max = order(7,5)
#
# print(max)
#
#
# #To get the values printed along with their index I can use Python's enumerate function like this
# mylist = ["a","b","c","d"]
# for i,j in enumerate(mylist):
#     print(i,j)
#
# # remove duplicate values
# myset=[1,2,3,1,2,3]
# myset2=[]
# for i in myset:
#     if not i in myset2:
#         myset2.append(i)
# print(myset2)
#
# #myset2=set(myset)
# #print(myset2)
#
# # ctrl+shift+f10 : run current file
# # ctrl+f10 : run last
#
# mydict={'a':'a','b':'b'}
#
# for mykey in mydict.keys():
#     print(mykey, end=' ')
#
# for myvalue in mydict.values():
#     print(myvalue, end=' ')
#
# for item1, item2 in mydict.items():
#     print(item1, item2)

# import matplotlib.pyplot as plt
#
# def lossf(x, y, w):
#     loss=0
#     for i in range(len(x)):
#         hf=w*x[i] #bias는 생략
#         loss += (hf-y[i])**2
#     return loss/len(x)
#
# x=[1,2,3]
# y=[1,2,3]
# w=10
# # print(lossf(x,y,w))
#
# x2,y2=[],[]
# for i in range(-50, 50):
#     w=i/10
#     loss=lossf(x,y,w)
#     # print(w, loss)
#     x2.append(w)
#     y2.append(loss)
#
# plt.plot(x2,y2)
# plt.plot(x2,y2,'ro')
# plt.show()

# x,y=map(int, input().split())
# print(x+1,y+1)

# import math as m
# print(m.sqrt(2))

# x=[0,2,4,6,8,1,3,5,7,9]
# print(x[len(x)-1])
# print(x[-1])
#
# print(x[-1:-(len(x)+1):-1])
#
# print([num*3 for num in range(1,5) if num % 2 == 0])
# print([x*y for x in range(2,10) for y in range(1,10)])


import numpy as np
# a = np.array([1,2,3])
# print(a, type(a))
# b=np.arange(5)
# print(b)
# c=np.arange(5,10,0.2)
# print(c)
#
# a+=1
# print(a)
#
# print(a[a<3])

# x=np.arange(15)
# print(x.shape)
# print(x)
# x2=x.reshape(5,3)
# print(x2)
# print(x2.shape)
# print()
# print(x2+2)
# # 2 2 2
# # 2 2 2
# # 2 2 2
# # 2 2 2
# # 2 2 2
# print(x2[0])
# print(x2[-1])
# print(x2[::-1,::-1])
# print(x2[2:4, 1:3])
# print(x2[0][1])
# print(x2.sum())
# print(x2.sum(axis=1))   #행 단위 합계
# print(x2.sum(axis=0))   #열 단위 합계

def sum(a, b):
    return a+b

def sum2(a,b):
    if type(a) != type(b):
        print("not possible")
        return
    else:
        res=sum(a,b)

    return res

if __name__ == "__main__":
    sum2('k',5)
    print(sum2(1,2))