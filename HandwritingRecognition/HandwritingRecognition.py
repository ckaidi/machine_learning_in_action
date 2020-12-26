from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

folderPath_traing='Data\\trainingDigits'
folderPath_test='Data\\testDigits'

#将图片转换为一个1*1024的向量，储存在文本文件中
def img2vector(fileName):
    returnVect=zeros((1,1024))
    fr=open(fileName)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+1]=int(lineStr[j])
    return returnVect

s=folderPath_traing+'\\0_13.txt'
print(img2vector(s))