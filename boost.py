import numpy as np
from numpy.core.fromnumeric import shape
from numpy.core.numeric import ones


#用于测试是否有某个值小于或者大于我们正在测试的阈值,所有在阈值一边的数据会分到类别-1
#而在另一边的数据会分到类别+1
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    #将返回数组的全部元素设置为1
    retArray=np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq=='lt':
        retArray[dataMatrix[:dimen]<=threshVal]=-1.0
    else:
        retArray[dataMatrix[:dimen]>threshVal]=1.0
    return retArray


#单层决策树生成函数，在一个加权数据集中循环，并找到具有最低错误率的单层决策树
#将最小错误率minError设为+∞
#  对数据集中的每一个特征(第一层循环)：
#    对每个步长(第二层循环)：
#      对每个不等号(第三层循环)：
#        建立一棵单层决策树并利用加权数据集对它进行测试
#        如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
#  返回最佳单层决策树
def buildStump(dataArr,classLabels,D):
    dataMatrix=np.matrix(dataArr)
    labelMat=np.matrix(classLabels).T
    m,n=shape(dataMatrix)#m是有多少个数据，n是每个数据集有多少个特征
    numSteps=10.0#用于在特征所有可能值上进行遍历
    bestStump={}#用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    bestClasEst=np.matrix(np.zeros((m,1)))
    minError=np.inf #错误率被初始化为无穷大,之后用于寻找最小的可能错误率
    #三层嵌套的for循环是程序的主要部分
    #第一层for循环在数据集的所有特征上循环，考虑数值型的特征，我们就可以通过计算最小值和最大值来了解需要多大步长
    for i in range(n):
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps
        #第二层循环再特征值上进行遍历
        for j in range(-1,int(numSteps)+1):
            #第三层循环在大于和小于之间切换不等式
            for inequal in ('lt','gt'):
                threshVal=(rangeMin+float(j)*stepSize)
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr=np.matrix(ones(m,1))#构建列向量，预测正确则将相应位置设置为0，错误则为1
                errArr[predictedVals==labelMat]=0
                weightedError=D.T*errArr#将错误向量errArr和权重向量的相应元素相乘并求和得到权重，这是AdaBoost和分类器交互的地方
                print ("split: dim %d ， thresh %.2 f ， thresh ineqal:%s ， the we ighted error is %.3f" %i,threshVal,inequal,weightedError)
                #如果当前错误率和已有最小错误率相比，如果当前值较小，那么就在字典bestStump中保存该单层决策树
                if weightedError<minError:
                    minError=weightedError
                    bestClasEst=predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    #最后返回字典，错误率和类别预测值
    return bestStump,minError,bestClasEst