# author='skyu'
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])#ravel()降为1维数组
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)#coutourf用等高线填充

    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y[0], cmap=plt.cm.Spectral)


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """


    s = 1 / (1 + np.exp(-x))

    return s


def load_planar_dataset():
    np.random.seed(1)
    '''
    seed( ) 用于指定随机数生成时所用算法开始的整数值，
    如果使用相同的seed( )值，则每次生成的随即数都相同，
    如果不设置这个值，则系统根据时间来自己选择这个值，
    此时每次生成的随机数因时间差异而不同。
    '''
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    #m/2 是个浮点数
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))   #range(start,end,add) 不靠括end，add默认为1
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        #np.linspace(start,end,num) [start,end] 均匀分布的num个实数,t.shape=(1,)

        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        t1=r * np.sin(t)
        t2=r * np.cos(t)
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)] #将切片对象沿第二个轴（按列）转换为连接。N*2的矩阵
        Y[ix] = j

    X = X.T#X.shape=200*m
    Y = Y.T#Y.shape=1*m

    return X, Y


def load_extra_datasets():
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2,
                                                                  n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)#[0,1)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure



'''
>>> np.random.seed(123)
>>> X = np.random.randint(0, 5, [3, 2, 2])
>>> print(X)

[[[5 2]
  [4 2]]

 [[1 3]
  [2 3]]

 [[1 1]
  [0 1]]]

>>> X.sum(axis=0)
array([[7, 6],
       [6, 6]])

>>> X.sum(axis=1)
array([[9, 4],
       [3, 6],
       [1, 2]])

>>> X.sum(axis=2)
array([[7, 6],
       [4, 5],
       [2, 1]])

'''