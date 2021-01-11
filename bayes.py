#词表到向量的转换函数
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec=[0,1,0,1,0,1]#1 代表侮辱性文字   0  代表正常言论
    return postingList,classVec

def createVocabList(dataSet):
    #创建一个空集
    vocabSet=set([])
    for doucument in dataSet:
        vocabSet=vocabSet|set(doucument)#创建两个集合的并集
    return list(vocabSet)


def setOfWords2Vec(vocabList,inputSet):
    #创建一个其中所含元素都为0的向量
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:print("the world:%s is not in my Vocabulary!"%word)
    return returnVec