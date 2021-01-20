import random
from numpy import *
from numpy.core.defchararray import multiply
from numpy.matrixlib.defmatrix import matrix


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

# i为第一个alpha的下标,m是所有alpha的数目。只要函数值等于输入值i，函数就会进行随机选择


def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j

# 用于调整大于H或小于L的alpha值


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

# 简化版SMO算法
#输入参数分别为:数据集，类别标签，常熟C，容错率，最大循环次数

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = matrix(dataMatIn)
    labelMat = matrix(classLabels).transpose()
    b = 0
    m, n = shape(dataMatrix)
    alphas = matrix(zeros(m, 1))
    iter = 0
    while(iter < maxIter):
        #用于记录alpha是否已经进行了优化
        alphaPairsChanged = 0
        for i in range(m):
            #fxi就是我们预测的类别
            fxi = float(multiply(alphas, labelMat).T *
                        (dataMatrix*dataMatrix[i, :].T))+b
            #计算预测类别和实际分类之间的误差
            Ei = fxi-float(labelMat[i])
            #如果alpha小于0或大于C时将被调整为0或C，所以他们一旦到达了边界就无需再优化
            if((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                #随机选择第二个alpha值，即alpha[j]
                j = selectJrand(i, m)
                fxj = float(multiply(alphas, labelMat).T *
                            (dataMatrix*dataMatrix[j, :].T))+b
                Ej = fxj-float(labelMat[j])
                #稍后要为新旧值变化进行对比，所以先要copy
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                #计算L,H,用于将alpha[j]调整到0到c之间
                if(labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j]-alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[j]+alphas[i]-C)
                    H = min(C, alphas[j]+alphas[i])
                if L == H:
                    print("L==H")
                    continue
                #eta是最优修改量
                eta = 2.0*dataMatrix[i, :]*dataMatrix[j, :].T-dataMatrix[i,
                                                                         :]*dataMatrix[i, :]-dataMatrix[j, :]*dataMatrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if(abs(alphas[j]-alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold-alphas[j])
                b1 = b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[i,
                                                                                        :].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i, :]*dataMatrix[j, :].T
                b2 = b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i, :]*dataMatrix[j,
                                                                                        :].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j, :]*dataMatrix[j, :].T
                if(0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif(0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1+b2)/2.0
                alphaPairsChanged += 1
                print("iter:%d i:%d,pairs changed %d %(iter,i,alphaPairsChanged")
        if(alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number:%d" % iter)
    return b, alphas
