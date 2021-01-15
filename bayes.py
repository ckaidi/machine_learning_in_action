from numpy import *
import re
import numpy as np
# 创建实验样本


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak',
                       'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 代表侮辱性文字   0  代表正常言论
    return postingList, classVec

# 创建一个包含在所有文档中出现的不重复词的列表


def createVocabList(dataSet):
    # 创建一个空集
    vocabSet = set([])
    for doucument in dataSet:
        vocabSet = vocabSet | set(doucument)  # 创建两个集合的并集
    return list(vocabSet)

# 输入参数位 词汇表和某个文档  输出文档向量
# 向量的每个元素为1或0，分别表示词汇表中的单词在输入文档中是否出现


def setOfWords2Vec(vocabList, inputSet):
    # 创建一个其中所含元素都为0的向量
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the world:%s is not in my Vocabulary!" % word)
    return returnVec

# 朴素贝叶斯分类器训练函数
# 输入参数:trainMatrix为文档的词汇表是否出现矩阵 trainCategory为每篇文档类别标签所构成的向量


def trainNBO(trainMatrix, trainCategory):
    # 文档的数量
    numTrainDocs = len(trainMatrix)
    # 词汇表的长度
    numWords = len(trainMatrix[0])
    # 是侮辱性文档的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 初始化概率
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        # 向量相加
        if trainCategory[i] == 1:
            # 所有词各自出现的次数
            p1Num += trainMatrix[i]
            # 所有词出现的总次数
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 为了解决多个很小的数相乘得到的结果下溢所以取自然对数
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

# 朴素贝叶斯分类函数


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0 = sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

# 测试函数


def testingNB():
    listOPosts, listClasses = loadDataSet()
    # 得到词汇表
    myVocablist = createVocabList(listOPosts)
    # 判断词汇表中的词在每个文档中是否出现
    trainMat = []
    for postingDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocablist, postingDoc))
    p0v, p1v, pAb = trainNBO(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocablist, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0v, p1v, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocablist, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0v, p1v, pAb))

# 词袋模型，上面的为词集模型，判断词汇表中的词是否在文档中出现
# 词袋模型判断词出现了几次


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


#文件解析及完整的垃圾邮件测试函数
def textParse(bigString):
    listOfTokens=re.split(r'\W+',bigString)#\W匹配任何非单词字符
    return [tok.lower() for tok in listOfTokens if(len(tok)>2)]

def spamTest():
    docList=[];classList=[];fullText=[]
    #读取26文件
    for i in range(1,26):
        wordList=textParse(open('DATA/Ch04/email/spam/%d.txt'%i).read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList=textParse(open('DATA/Ch04/email/ham/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    trainingSet=[i for i in range(50)];testSet=[]
    #随机构建训练集
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        #删除变量对数据的引用，而不是删除数据本身
        #避免出现重复数据
        del(trainingSet[randIndex])
    trainMat=[];trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0v,p1v,pSpam=trainNBO(array(trainMat),array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if(classifyNB(array(wordVector),p0v,p1v,pSpam)!=classList[docIndex]):
            errorCount+=1
    print ('the error rate is:',float(errorCount)/len(testSet)*100,'%')

# main
spamTest()