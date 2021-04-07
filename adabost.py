from math import exp
from boost import buildStump, stumpClassify
import numpy as np
from numpy.core.fromnumeric import shape
from numpy.core.numeric import ones
import matplotlib.pyplot as plt

def loadSimpData():
    dataMat=np.martix([[1.,2.1],
    [2.,3.1],
    [1.3,1.],
    [1.,1.]
    [2.,1.]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

#对每次迭代:
#  利用buildStump函数找到最佳单层决策树
#  将最佳单层决策树加入到单层决策树数组
#  计算alpha
#  计算新的权重向量D,包含了每个数据点的权重
#  更新累计类别估计值
#  如果错误率等于0.0，则退出循环
def adaBoostTrainDS(dataArr,classLabels,numIt=40):#输入参数：数据集、类别标签、迭代次数
    weakClassArr=[]
    m=shape(dataArr)[0] #数据集的数量
    D=np.matrix(ones(m,1)/m) #每个数据点的权重初始化为1
    aggClassEst=np.matrix(np.zeros(m,1)) #记录每个数据点的类别估计累计值
    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataArr,classLabels,D) #建立单层决策树
        print('D:',D.T)
        alpha=float(0.5*np.log((1.0-error)/max(error,1e-16))) #alpha会告诉总分类器本次单层决策树输出结果的权重
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)
        print('classEst:',classEst.T)
        expon=np.multiply(-1*alpha*np.matrix(classLabels).T,classEst)
        D=np.multiply(D,exp(expon))
        D=D/D.sum()
        aggClassEst+=alpha*classEst
        print('aggClassEst:',aggClassEst.T)
        aggErrors=np.multiply(np.sign(aggClassEst)!=np.matrix(classLabels).T,ones((m,1)))
        errorRate=aggErrors.sum()/m
        print('total error:',errorRate,'\n')
        if errorRate==0.0:break
    return weakClassArr

#利用弱分类器对数据进行分类
#输入参数：一个或者多个待分类的样例，多个弱分类器组成的数组
def adaClassify(datToClass,classifierArr):
    dataMatrix=np.matrix(datToClass)
    m=shape(dataMatrix)[0]
    aggClassEst=np.matrix(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return np.sign(aggClassEst) #sign函数用来判断数据符号

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#ROC曲线的绘制及AUC计算函数
#第一个参数是numpy数组或者一个行向量组成的矩阵，该参数代表的是分类器预测的强度
#第二个参数是classLabels
def plotROC(predStrengths,classLabels):
    cur=(1.0,1.0)
    ySum=0.0
    numPosClas=sum(np.array(classLabels)==1.0)
    yStep=1/float(numPosClas)
    xStep=1/float(len(classLabels)-numPosClas)
    sortedIndicies=predStrengths.argsort()
    fig=plt.figure()
    fig.clf()
    ax=plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index]==1.0:
            delX=0;delY=yStep
        else:
            delX=xStep;delY=0
            ySum+=cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
        cur=([cur[0]-delX,cur[1]-delY])
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is:",ySum*xStep)