import numpy as np
from numpy.core.fromnumeric import shape
from numpy.core.numeric import ones


#Logistic回归梯度上升优化算法
def loadDataSet():
    dataMat=[];labelMat=[]
    fr=open('DATA/Ch05/testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split()
        #三个参数，第一个是回归系数，第二个是特征值一，第三个是特征值二
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
        return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

#梯度算法主要步骤
#dataMatIn:2维的Numpy数组，每列分别代表每个不同的特征，每行代表每个训练样本
def gradAscent(dataMatIn,classLabels):
    #转换为numpy的矩阵数据类型
    dataMatrix=np.mat(dataMatIn)
    labelMat=np.mat(classLabels)
    m,n=shape(dataMatrix)
    #alpha是向目标移动的步长
    alpha=0.01
    #maxCycles是迭代次数
    maxCycles=500
    weights=ones((n,1))
    #计算真是类别和预测类别的差值,按照差值的方向调整回归系数
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights