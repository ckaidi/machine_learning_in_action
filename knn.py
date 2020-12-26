from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

#为避免某一个数据严重影响计算结果，我们需要把所有数据映射到[0,1]或者[-1,1]
def autoNorm(dataSet):
    minVals=dataSet.min(0)

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


datingDataMat,labels=file2matrix('datingTestSet2.txt')
fig=plt.figure()
ax=fig.add_subplot(111)
#后面两个参数赋予了颜色
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(labels),15.0*array(labels))
plt.show()