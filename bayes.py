from numpy import *

#创建实验样本
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec=[0,1,0,1,0,1]#1 代表侮辱性文字   0  代表正常言论
    return postingList,classVec

#创建一个包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
    #创建一个空集
    vocabSet=set([])
    for doucument in dataSet:
        vocabSet=vocabSet|set(doucument)#创建两个集合的并集
    return list(vocabSet)

#输入参数位 词汇表和某个文档  输出文档向量
#向量的每个元素为1或0，分别表示词汇表中的单词在输入文档中是否出现
def setOfWords2Vec(vocabList,inputSet):
    #创建一个其中所含元素都为0的向量
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:print("the world:%s is not in my Vocabulary!"%word)
    return returnVec

#朴素贝叶斯分类器训练函数
#输入参数:trainMatrix为文档的词汇表是否出现矩阵 trainCategory为每篇文档类别标签所构成的向量
def trainNBO(trainMatrix,trainCategory):
    #文档的数量
    numTrainDocs=len(trainMatrix)
    #词汇表的长度
    numWords=len(trainMatrix[0])
    #是侮辱性文档的概率
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    #初始化概率
    p0Num=zeros(numWords);p1Num=zeros(numWords)
    p0Denom=0.0;p1Denom=0.0
    for i in range(numTrainDocs):
        #向量相加
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=p1Num/p1Denom
    p0Vect=p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive