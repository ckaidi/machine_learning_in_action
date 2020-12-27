from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

filePath='trainingData/datingTestSet2.txt'

def classifyPerson():
    resultList=['not at all','in small does','in large does']
    percentTats=float(input("percentage of time spent playing video games?"))
    flyingMiles=float(input("frequent flier miles earned per year?"))
    icecream=float(input("liters of iec cream consumed per year?"))
    datingDataMat,datingLabels=file2matrix(filePath)
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([flyingMiles,percentTats,icecream])
    result=classify0(inArr,datingDataMat,datingLabels,3)
    print("You will probably like this person:"+resultList[result-1])

def datingClassTest():
    #测试数据的比例，剩下来的都是训练样本
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix(filePath)
    normMat,ranges,minVals=autoNorm(datingDataMat)
    #数据的行数
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        # normMat[i,:]进行的是切片操作，把第i行的所有数据取出，normMat[i,0:2]表示把第i行的第一，第二个数据取出
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with:%d,the real answer is %d"%(classifierResult,datingLabels[i]))
        if(classifierResult!=datingLabels[i]): errorCount+=1.0
    print("the total error rate is：%f"%(errorCount/float(numTestVecs)))

#为避免某一个数据严重影响计算结果，我们需要把所有数据映射到[0,1]或者[-1,1]
def autoNorm(dataSet):
    #每列的最小值放在变量minVals中，将最大值放在maxVals中，其中min(0)中的参数0使得函数可以从列中选取最小值
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    range=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(range,(m,1))
    return normDataSet,range,minVals

#得到特征列表和标签列表
def file2matrix(filename):
    fr=open(filename)
    arrayOfLines=fr.readlines()
    #得到文件行数
    numberOfLines=len(arrayOfLines)
    #创建返回的Numpy矩阵
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOfLines:
        #去除首尾的空格键
        line=line.strip('\n')
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    label=['A','A','B','B']
    return group,label

#inX:输入向量   dataSet:训练样本集   labels:标签向量   k:表示用于选择最近邻居的数目#
def classify0(inX,dataSet,labels,k):
    #训练样本集的行数
    dataSize=dataSet.shape[0]
    #将inX在(dataSize,1)维度上重复
    diffMat=tile(inX,(dataSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    #从小到大排列数组并返回他们的索引数组
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),reverse=True)
    return sortedClassCount[0][0]

"""datingDataMat,labels=file2matrix(filePath)
fig=plt.figure()
ax=fig.add_subplot(111)
#后面两个参数赋予了颜色
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(labels),15.0*array(labels))
plt.show()"""
