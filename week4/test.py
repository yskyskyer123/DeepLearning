# author='skyu'
# -*- coding: utf-8 -*-


import numpy as np



def run():
    global xx
    xx=3
if __name__=='__main__':
    # a=np.array([1,2,3])
    # b=np.array(a)
    # b[2]=4
    #
    # print(a,b)
    #
    # a = np.array([1, 2, 3])
    # b = np.array(a,copy=False)
    # b[2] = 4
    #
    # print(a, b)
    #
    # a = np.array([1, 2, 3])
    # b = np.array(a, copy=True)
    # b[2] = 4
    #
    # print(a, b)
    #
    #
    # x=np.array([1,2,3,4,5])
    # print(x[2])
    #
    # y = np.array([[1, 2, 3, 4, 5] ])
    # # print(y[2])
    # print("p1=",y[0][2])
    # print("p2=",y[0,2])
    # print("p3=",y[(0,2)])
    #
    #
    # x=x.reshape(5,1)
    # print(x)
    #
    # x = x.reshape(1, 5)
    # print(x)
    #
    # x = x.reshape(5,)
    # print(x)
    #
    # x = x.reshape(5)
    # print(x)

    # run()
    # print(xx)
    ar=np.array([1,2,3,4,3])
    br=np.where(ar==3)
    print(br)
    cr=np.asarray(br)
    print(cr)


    dr=np.array([[ 2.,  2.,  2.,  2.,  2.,  1.,  2.,  2.,  2.,  2.,  2.,  1.,  2.,  1.,  0.,  2.,  0.,  2,
   2.,  1.,  2.,  0.,  0.,  2.,  2.,  2.,  2.,  0.,  1.,  1.,  2.,  2.,  2.,  2.,  1.,  0.,
   0.,  2.,  1.,  0.,  2.,  2.,  2.,  0.,  1.,  1.,  2.,  2.,  2.,  0.]])
    print(np.where(dr==1) )

