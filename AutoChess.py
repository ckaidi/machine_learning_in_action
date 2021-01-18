from numpy import *
import operator

import numpy
from datingPersonClassify import classify0

fileTraining='AutoChessData/0.txt'

def file2matrix():
    fr=open(fileTraining)
    arrayOfLines=fr.readlines()
    #得到文件行数
    numberOfLines=len(arrayOfLines)
    #创建返回的Numpy矩阵
    returnMat=zeros((numberOfLines,8))
    classLabelVector=[]
    index=0
    for line in arrayOfLines:
        #去除首尾的空格键
        line=line.strip('\n')
        listFromLine=line.split(',')
        returnMat[index,:]=listFromLine[0:8]
        classLabelVector.append(listFromLine[-1])
        index+=1
    return returnMat,classLabelVector

def createMatrix(datSet):
    dataMatrix=[]
    for i in range(len(datSet)):
        eachGame=datSet[i]
        gameData=zeros(12)
        for num in eachGame:
            gameData[int(num)]+=1
        dataMatrix.append(gameData)
    return dataMatrix

 #inX:输入向量   dataSet:训练样本集   labels:标签向量   k:表示用于选择最近邻居的数目#   
def classify0(inX,dataSet,labels,k):
    classes=set(labels)
    #训练样本集的行数
    dataSize=dataSet.shape[0]
    #将inX在(dataSize,1)维度上重复
    diffMat=tile(inX,(dataSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5

    classDis={}
    classNum={}
    for i in range(len(distances)):
        classDis[labels[i]]=classDis.get(labels[i],1)+distances[i]
        classNum[labels[i]]=classNum.get(labels[i],1)+1
    for label in classes:
        classDis[label]=classDis.get(label,1)/classNum.get(label,1)
    
  
    sortedClassCount=sorted(classDis.items(),key=lambda item:item[1])
    return sortedClassCount[0:k]
#main
a,b=file2matrix()
c=createMatrix(a)
testVect = input()
testMat=testVect.split(',')
dataTest=zeros(12)
for i in testMat:
    dataTest[int(i)]+=1
result=classify0(dataTest,numpy.array(c),b,10)
print(result)