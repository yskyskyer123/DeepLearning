# author='skyu'
# -*- coding: utf-8 -*-



import numpy as np



a=np.random.randn(2,3)
b=np.random.randn(2,1)
c=a+b
# numpy中有一些常用的用来产生随机数的函数，randn()和rand()就属于这其中。
# numpy.random.randn(d0, d1, …, dn)是从标准正态分布中返回一个或多个样本值。
# numpy.random.rand(d0, d1, …, dn)的随机样本位于[0, 1)中。

print(str(a)+"\n"+str(b)+"\n"+str(c)+"\n")



a=np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
b=np.array([[2]])
#
# a = np.random.randn(4, 3) # a.shape = (4, 3)
# b = np.random.randn(4, 1) # b.shape = (3, 2)
c = a*b
print(c)