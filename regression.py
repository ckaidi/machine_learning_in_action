from numpy import Inf, exp, inf, zeros
from numpy.core.fromnumeric import mean, shape, var
from numpy.lib import eye
from numpy.linalg.linalg import tensorsolve
import numpy.linalg.linalg as linalg
from numpy.matrixlib.defmatrix import matrix
from time import sleep
import json
import random
import urllib

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#用于计算最佳拟合曲线
def standRegers(xArr,yArr):
    xMat=matrix(xArr);yMat=matrix(yArr).T
    xTx=xMat.T*xMat
    #判断是否可逆，det求矩阵行列式，行列式为0则不可逆
    if linalg.det(xTx)==0.0:
        print ("This matrix is singular,cannot do inverse")
        return
    ws=xTx.I*(xMat.T*yMat)
    return ws

#局部加权线性回归函数
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat=matrix(xArr);yMat=matrix(yArr).T
    m=shape(xMat)[0]
    #创建单元矩阵
    weights=matrix(eye(m))
    for j in range(m):
        #权重值大小以指数级衰减
        diffMat=testPoint-xMat[j,:]
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx)==0:
        print("This matrix is singular,cannot do inverse")
        return
    ws=xTx.T*(xMat.T*(weights*yMat))
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=shape(testArr)[0]
    yHat=zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat

def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

#岭回归(岭回归就是在矩阵xTx上加入一个入I从而使得矩阵非矩阵)
#该函数用于计算回归系数
#实现了在给定的lamba下的岭回归求解
#lam应该以指数值变化，这样就可以看出lambda在很大和很小的时候对结果造成的影响
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat
    denom=xTx+eye(shape(xMat)[1])*lam #eye生成对角线上全是1，其余位置都是0的矩阵(单位矩阵)
    #在一般的回归中，行列式为0会出错，在岭回归中，如果lamba为0，则一样会出现错误，所以最后一样要判断矩阵是否为非奇异
    if linalg.det(denom)==0.0: 
        print('This matrix is singular cannot do inverse')
        return
    ws=denom.I*(xMat.T*yMat)
    return ws


#函数ridgeTest()用于在一组上测试结果
def ridgeTest(xArr,yArr):
    xMat=matrix(xArr);yMat=matrix(yArr).T
    yMean=mean(yMat,0)      #数据标准化处理
    yMat=yMat-yMean         #使每项特征都具有相同的重要性
    xMean=mean(xMat,0)      #具体做法就是所有特征值都减去各自的均值并除夕方差
    xVar=var(xMat,0)        #
    xMat=(xMat-xMean)/xVar  #
    numTestPts=30
    wMat=zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws=ridgeRegres(xMat,yMat,exp(i,-10))
        wMat[i,:]=ws.T
    return wMat


#矩阵归一化
#为了让每一维数据处在同一个度量单位，相当于消除量纲的作用。
#将各行第1个非零元素化成1（该行除以这个非零元素）；是将各行向量，单位化，也即各行除以该行的行向量的模的开方
#假设行向量是(a，b，c，d)
#化成(a/√(a²+b²+c²+d²)，b/√(a²+b²+c²+d²)，c/√(a²+b²+c²+d²)，d/√(a²+b²+c²+d²))
def regularize(xMat):  #regularize by colums
    inMat=xMat.copy()
    inMeans=mean(inMat,0)
    inVar=var(inMat,0)
    inMat=(inMat-inMeans)/inVar
    return inMat


#前向逐步回归,属于一种贪心算法，每一步都尽可能减少误差
#伪代码
#数据标准化,使其分布满足0均值和单位方差
#在每轮迭代过程中:
#  设置当前最小误差lowestError为正无穷
#  对每个特征：
#    增大或缩小：
#      改变一个系数得到一个新的W
#      计算新W下的误差
#      如果误差Error小于当前最小误差lowestError:设置Wbest等于当前的W
#    将W设置为新的Wbest
def stageWise(xArr,yArr,eps=0.01,numIt=100): #xArr为输入数据，yArr为预测数据，eps为每次迭代需要调整的步长
    xMat=matrix(xArr);yMat=matrix(yArr).T
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMat=regularize(xMat)
    m,n=shape(xMat) #mw为数据的数量，n为每个数据特征的数量
    returnMat=zeros((numIt,n))
    ws=zeros((n,1));wsTest=ws.copy();wsMax=ws.copy
    for i in range(numIt):
        print(ws.T)
        lowestError=Inf
        for j in range(n):
            for sign in[-1,1]: #增大或减小
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=rssError(yMat.A,yTest.A)
                if rssE<lowestError:
                    lowestError=rssE
                    wsMax=wsTest
        ws=wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat


#利用缩减法确定最佳回归系数
#交叉验证测试岭回归
def crossVaildation(xArr,yArr,numVal=10):
    m=len(yArr)
    indexList=range(m)
    errorMat=zeros((numVal,30))
    #创建训练集和测试集容器
    for i in range(numVal):
        trainX=[];trainY=[]
        testX=[];testY=[]
        random.shuffle(indexList) #shuffle()函数将序列的元素随机排序
        for j in range(m):
            if j <m*0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat=ridgeTest(trainX,trainY)
        for k in range(30):
            #用训练时的参数将测试数据标准化
            matTestX=matrix(testX);matTrainX=matrix(trainX)
            meanTrain=mean(matTrainX,0)
            varTrain=var(matTrainX,0) #var按行或按列求平均，没有参数则是所有求平均
            matTestX=(matTestX-meanTrain)/varTrain
            


#------------------------------------------------main----------------------------------------------------#
'''
dataMat,labelMat=loadDataSet('D:/毕业设计/DATA/43f/43features_18.txt')
#dataMat,labelMat=loadDataSet('DATA/Ch08/abalone.txt')
#yHat01=lwlrTest(dataMat[0:1999],dataMat[0:1999],labelMat[0:1999],0.1)
#yHat1=lwlrTest(dataMat[0:1999],dataMat[0:1999],labelMat[0:1999],1)
#yHat10=lwlrTest(dataMat[0:1999],dataMat[0:1999],labelMat[0:1999],10)
#print(rssError(labelMat[0:1999],yHat01.T))
#print(rssError(labelMat[0:1999],yHat1.T))
#print(rssError(labelMat[0:1999],yHat10.T))
ws=standRegers(dataMat,labelMat)
s=shape(ws)
ws=ws.tolist()
fr=open("ws_18.txt","w+")
for i in ws:
    fr.write(str(i[0]))
    fr.write("\t")
fr.close()
print(s)
print(ws)
'''
a=zeros((3,1))
print(a)