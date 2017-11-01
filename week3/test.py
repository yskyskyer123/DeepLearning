# author='skyu'
# -*- coding: utf-8 -*-

import numpy as np
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
import matplotlib.pyplot as plt



if __name__=='__main__':
    a=np.array([ [1,2], [3,4] ])
    b=np.sum(a,axis=0)#不加keepdims=True 会丢失数量为1的维度
    print(b)
    c=np.sum(a,axis=0,keepdims=True)
    print(c)
    # X, Y = load_planar_dataset()

    # m=4
    # m2=m/2
    # n=int(m/2)
    # x=np.array([[1,2],[3,4]])
    # print(x-3)
    #
    # plt.figure(figsize=(9, 6))
    # n = 1000
    # # rand 均匀分布和 randn高斯分布
    # x = np.random.randn(1, n)
    # y = np.random.randn(1, n)
    # T = np.arctan2(x, y)
    # print(x.shape,y.shape,T.shape)
    # plt.scatter(x, y, c=T, s=25, alpha=0.4, marker='o')
    # # T:散点的颜色
    # # s：散点的大小
    # # alpha:是透明程度
    # plt.show()
    #
    # x=np.array([2,4])
    # print(x.shape)
    #
    #
    # x=np.array([[1,2,3],[4,5,6]])
    # print(np.sum(x))
    # print(round(0.5) )
    # print(round(0.51) )
    # print(round(0.49) )


    x=np.array([1 , 2, 3, 4])
    print(x)
    y=np.array([10,11,12,13,14])
    print(y)
    xx,yy=z=np.meshgrid(x,y)
    print(z)
    print(xx.ravel())
    print(yy.ravel())

    z=np.array([ [1,2,3,4],[5,6,7,8]])
    print(z.size)
    z2=np.array([1,2,3,4])
    print(z2.size)

    print('##########################')
    x=np.array([ [1,1,1,1],[1,1,1,1],[1,1,1,1]])
    print(np.sum(x,axis=0))
    print(np.sum(x,axis=0,keepdims=True))






