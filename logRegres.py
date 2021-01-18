from os import error
import numpy as np
import matplotlib.pyplot as plt
from numpy import *


#Logistic回归梯度上升优化算法
def loadDataSet():
    dataMat=[];labelMat=[]
    fr=open('DATA/Ch05/testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        #三个参数，第一个是回归系数，第二个是特征值一，第三个是特征值二
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

#梯度算法主要步骤
#dataMatIn:2维的Numpy数组，每列分别代表每个不同的特征，每行代表每个训练样本
def gradAscent(dataMatIn,classLabels):
    #转换为numpy的矩阵数据类型
    dataMatrix=np.mat(dataMatIn)
    #transpose把矩阵中的行和列位置对调
    labelMat=np.mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    #alpha是向目标移动的步长
    alpha=0.001
    #maxCycles是迭代次数
    maxCycles=500
    weights=ones((n,1))
    #计算真是类别和预测类别的差值,按照差值的方向调整回归系数
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights

#画出数据及和Logistic回归最佳拟合直线的函数
def plotBestFit(weights):
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n=shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if(int(labelMat[i])==1):
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    #fig=plt.figure()
    plt.subplot(111)
    plt.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    plt.scatter(xcord2,ycord2,s=30,c='green')
    x=np.arange(-3.0,3.0,0.1)
    #设置sigmoid函数为0，0是两个分类的分界处，因此设定0=w0x0+w1x1+w2x2然后解出X2和X1的关系式，即分隔线方程，注意X0=1
    #y=(-weights[0]-weights[1]*x)/weights[2])
    y=[]
    for i in range(len(x)):
        y.append(float((-weights[0]-weights[1]*x[i])/weights[2]))
    #ax.plot(x,y)
    plt.plot(x,y,'-b')
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()

#随机梯度上升算法
def stocGrandAscent0(dataMatrix,classLabels):
    m,n=shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return weights

#改进的随机梯度上升算法,numIter为迭代次数
def stocGrandAscent1(dataMatrix,classLabels,numIter=150):
    m,n=shape(dataMatrix)
    weights=ones(n)
    for j in range(numIter):
        dataIndex=range(m)
        for i in range(m):
            #可以适当加大常数项来确保新的值获得更大的回归系数
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

#-----main-----#
dataArr,labelMat=loadDataSet()
weights=gradAscent(dataArr,labelMat)
print(weights)
plotBestFit(weights)