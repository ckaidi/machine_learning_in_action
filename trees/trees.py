from math import log


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
        #遍历当前特征中的所有唯一属性值，为每个特征划分依次数据集
        for value in uniqueVals:
            # 不同特征类型，不同特征值的返回数据集
            subDataSet = splitDataSet(dataSet, i, value)
            #该特征的数据集在所有数据中的比例
            prob = len(subDataSet)/float(len(dataSet))
            #把按某一类特征所有特征值分类的数据集的熵相加进行比较
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        #计算最好的信息熵
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
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


myDat, labels = createDataSet()
reDataSet = splitDataSet(myDat, 0, 1)
print(reDataSet)
