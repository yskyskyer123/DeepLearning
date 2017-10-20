# author='skyu'
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import math


#返回一个矩阵，由m_train个列向量组成，每个列向量表示了一幅图片，图片里面元素的排列时是R，G，B，R,G,B，...,R，G，B
def my_reverse(set_x):
    return set_x.reshape(set_x.shape[0],-1).T
    '''
    X.reshape(na,-1) 将X分为na组，不改变元素顺序
    X.T     即为X的转置，X的shape[0]和shape[1]互换，改变了X中元素的顺序
    此时X的列数正好是图片个数，每一个列向量代表一幅图片的信息

    '''

def sigmoid(z):
    s = 1/(1+np.exp(-z))        #用math.exp报错,math.exp智能处理单个数，np.exp处理矩阵/向量
    return s

#初始化w和b，w应该是一个nx*1的矩阵
def initialize_with_zeros(dim):

    w = np.zeros( (dim,1) )
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w,b


def propagate(w, b, X, Y):

    m = X.shape[1]

    A = sigmoid( np.dot(w.T,X)+b)
    cost =    -1./m*( np.dot( Y,np.log(A.T) )+np.dot( 1-Y , np.log( (1-A).T )  )  )  # 计算的结果仍然是一个矩阵比如[[ 6.00006477]]

    dz=A-Y

    dw = 1./m*( np.dot(X,dz.T)) #对于一组训练数据 dw[i]=dz*x[i], 对于m组  dw[i]=1/m( dz[0]*x[0][i]+dz[1]*x[1][j]+...dz[m-1]*x[m-1][i] )
    db = 1./m*np.sum(dz)        #对于一组训练数据 db=dz,         对于m组  dz=1/m(dz[0]+dz[1]+...+dz[m-1])

    assert (dw.shape == w.shape)
    assert (db.dtype == float)

    cost = np.squeeze(cost)         #将一维的去掉
    assert (cost.shape == ())       #最后变成一个数

    grads = {"dw": dw,  "db": db}    #注意字典的用法

    return grads, cost

# GRADED FUNCTION: optimize

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        #梯度下降算法
        w = w-learning_rate*dw
        b = b-learning_rate*db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs



global eps
eps= 1e-10


def dcmp(x):
    if math.fabs(x)<eps:
        return 0
    return -1 if x<0 else 1

def predict(w, b, X):

    m = X.shape[1]

    Y_prediction = np.zeros((1, m),dtype=int)

    w = w.reshape(X.shape[0], 1)#感觉没有必要

    A =  np.dot(w.T,X)+b


    for i in range(A.shape[1]):
        Y_prediction[0,i]= 0 if  dcmp(A[0,i]-0.5)<=0 else 1

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)

    ### END CODE HERE ###
    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
     "Y_prediction_test": Y_prediction_test,
     "Y_prediction_train": Y_prediction_train,
     "w": w,
     "b": b,
     "learning_rate": learning_rate,
     "num_iterations": num_iterations}

    return d



def run():

    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,print_cost=True)

    costs = np.squeeze(d['costs'])#感觉没用   costs本来就是一个列表

    # t = np.arange(-1, 2, 0.15)
    # plt.plot(t,costs)#t中必须有和costs相同的元素数量，图像形状由costs决定，和t中值无关

    plt.plot(costs)#会根据costs里面的20个元素来绘制
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()


def test_learning_rate():

    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    for i in learning_rates:
        print("learning rate is: " + str(i))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i,
                               print_cost=True)
        print('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))


    plt.ylabel('cost')
    plt.xlabel('iterations')


    legend = plt.legend(loc='upper center', shadow=True)#图例,有了这句话可以显示图例
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


def test_my_image():

    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,
              print_cost=True)

    my_image = "my_image.jpg"


    fname = "images/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    #flatten : bool, optional
    #If True, flattens the color layers into a single gray-scale layer.


    my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
    my_predicted_image = predict(d["w"], d["b"], my_image)

    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
        int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")



#用于显示读取的图像
def show_image():

    #计算分成多少行列显示，n rows m colomus
    m=int(math.sqrt(m_train))
    n=  m_train/m if m_train%m==0 else int(m_train/m)+1

    for i in range(m_train):
        plt.subplot(  n,m ,1 + i)         #subplot分成多个小窗口显示
        plt.imshow(train_set_x_orig[i])   #imshow显示图像，plt.plot绘制曲线
    plt.show()                            #将绘制的结果显示出来


if __name__=='__main__':

    # Loading the data (cat/non-cat)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()#y是行向量


    #Example of a picture
    # index = 25
    # plt.imshow(train_set_x_orig[index])
    #
    # print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
    #set_y的shape是1*m_train，必须以二维形式访问
    #squeeze把shape为1的维度去掉
    #classes就相当于map

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]


    # Reshape the training and test examples


    train_set_x_flatten=my_reverse(train_set_x_orig)
    test_set_x_flatten=my_reverse(test_set_x_orig)

    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    #run()
    test_learning_rate()
    # test_my_image()


