# author='skyu'
# -*- coding: utf-8 -*-


import h5py
import numpy as np

def load():

    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    # print (train_dataset.keys())




def fun():
    fun_num1=3
def fun2():
    print(fun_num1)
def runn():
    a=x+1
    print(a)
    print(t)
if __name__=='__main__':
    load()
    a=[1 ,2 ,3,4,5,6]
    b=np.array(a[:])
    b2=np.array(a)
    print(b)
    print(b2)
    c=(100,)
    d=(100)
    print(c)
    print(d)

    ar=[
        [[1,2],[3,4],[5,6],[7,8]],
        [[21,22], [23, 24], [25, 26], [27, 28]],
        [[31, 32], [33, 34], [35, 36], [37, 38]]

    ]
    ar=np.array(ar)
    print(ar.shape)
    br=ar
    na,nb,nc=ar.shape
    br=br.reshape(nc,-1)
    print(br)
    cr=br.T
    print(cr)

    assert(2==2)


    xx=np.zeros((3,1))
    print(xx.shape,xx.dtype)

    x=[[2,3],[4,5]]
    print(x[1][1])


    x=[[1,2] ]
    print(np.squeeze(x))

    import numpy as np
    import matplotlib.pyplot as plt

    t = np.arange(-1, 2, .01)
    s = np.sin(2 * np.pi * t)
    s2 = np.cos(2 * np.pi * t)

    plt.plot(t, s,label='sin')
    plt.plot(t,s2,label='cos')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # draw a thick red hline at y=0 that spans the xrange
    l = plt.axhline(linewidth=4, color='r')
    plt.axis([-1, 2, -1, 2])
    plt.show()
    plt.close()

    x=3;
    # runn()
    fun()
    # fun2()
    print(3/2)
    print(3%2)
    print(int(1.4))
    print(int(1.5))
    print(int(1.6))