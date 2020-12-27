from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
from datingPersonClassify import classify0

folderPath_traing='DATA/Ch02\handWriteRe/trainingDigits'
folderPath_test='DATA/Ch02/handWriteRe/testDigits'


def handwritingClassTest():
    hwLabels=[]
    #获取训练文件列表
    trainingFileList=listdir(folderPath_traing)
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumberStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumberStr)
        trainingMat[i,:]=img2vector(folderPath_traing+'/'+fileNameStr)
    testFileList=listdir(folderPath_test)
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumberStr=int(fileStr.split('_')[0])
        s=folderPath_test+'/'+fileNameStr
        vectorUnderTest=img2vector(folderPath_test+'/'+fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with:%d,the real answer is:%d"%(classifierResult,classNumberStr))
        if(classifierResult!=classNumberStr):errorCount+=1.0
    print("\nthe total number of errors is:%d"%errorCount)
    print("\nthe total error rate is:%f"%(errorCount/float(mTest)))

#将图片转换为一个1*1024的向量，储存在文本文件中
def img2vector(fileName):
    returnVect=zeros((1,1024))
    fr=open(fileName)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=lineStr[j]
    return returnVect


handwritingClassTest()
