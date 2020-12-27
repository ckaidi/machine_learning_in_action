from math import log
import operator


dataPath='DATA\Ch02\datingTestSet.txt'

#递归构造决策树
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    #类别完全相同则停止继续划分
    #count() 方法用于统计字符串里某个字符出现的次数
    if(classList.count(classList[0])==len(classList)):
        return classList[0]
    #遍历完所有特征时返回出现次数最多的
    if(len(dataSet[0])==1):
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    #del用于list列表操作，删除一个或者连续几个元素
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
    
# classList为分类名称列表，
# classCount存储了classList中每个类标签出现频率，返回出现最多次数最多的分类名称
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    #operator模块提供的itemgetter函数用于获取对象的哪些维的数据
    sortedClassCount=sorted(classCount.items,key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def chooseBestFeatureToSplit(dataSet):
    # 去掉最后一个标签值
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 将dataSet中的数据先按行依次放入example中，然后取得example中的example[i]元素，放入列表featList中
        # 即把每个特征值依次取出来计算
        featList = [example[i] for example in dataSet]
        # 创建一个无序不重复的元素集
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 遍历当前特征中的所有唯一属性值，为每个特征划分依次数据集
        for value in uniqueVals:
            # 不同特征类型，不同特征值的返回数据集
            subDataSet = splitDataSet(dataSet, i, value)
            # 该特征的数据集在所有数据中的比例
            prob = len(subDataSet)/float(len(dataSet))
            # 把按某一类特征所有特征值分类的数据集的熵相加进行比较
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy-newEntropy
        # 计算最好的信息熵
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 按照给定特征划分数据集,输入参数分别是:待划分的数据集、划分数据集的特征索引、返回拥有该特征值的数据集
def splitDataSet(dataSet, axis, value):
    reDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 接下来的两个语句是为了把特征值去掉
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            reDataSet.append(reducedFeatVec)
    return reDataSet


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no'], ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        # log(x,y)表示以y为底求对数
        shannonEnt -= prob*log(prob, 2)
    # 熵越高说明混合的数据越多
    return shannonEnt

