import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_arge = dict(arrowstyle="<-")
dataTest = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
            {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]


# 在父子节点填充文本信息
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt=(plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD
    for key in secondDict.keys():
        if(type(secondDict[key]).__name__=='dict'):
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD


# 获取叶节点的数目
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if(type(secondDict[key]).__name__ == 'dict'):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 获取树的层数
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if(type(secondDict[key]).__name__ == 'dict'):
            thisDepth = 1+getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if(thisDepth > maxDepth):
            maxDepth = thisDepth
    return maxDepth


# 绘制带箭头的注解
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    #：在改程序中使用了变量createPlot.ax1这个变量，这是一种全局变量的表达方式。
    # 在两个子函数下的变量本来都是局部变量，不能跨出变量的作用域使用，此处使用“函数名.变量名”的方式，将变量申明为全局变量。
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_arge)


def createPlot(inTree):
    #figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, 
    # frameon=True, FigureClass=<class 'matplotlib.figure.Figure'>, clear=False, **kwargs)
    #num(optional):可以理解为该窗口的id,即该窗口的身份标识
    #figsize(optional):整数元组，例如(4,4)即以长4英寸,宽4英寸的大小创建窗口
    #dpi(optional):整数,表示该窗口分辨率
    #facecolor(optional):表示窗口背景的颜色,颜色设置通过RGB,范围是'#000000'~'#FFFFFF',其中每2个字节16位表示RGB的0~255
    #edgecolor(optional):表示窗口的边框颜色
    #frameon(optional):表示是否绘制窗口的图框
    #FigureClass(optional):图形的子类可以选择使用自定义地物实例。
    #clear(optional):如果是True,并且图形已经存在，则清除该图形 
    #*args和**kwargs都代表1个或多个参数的意思,*args传入tuple类型的无名参数
    #而**kwargs传入的参数是dict类型
    fig = plt.figure()
    #Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    #其作用是把一个绘图区域分为多个子区域，并把需要绘制出来的图画在分好的指定区域内。
    #函数的意思是将整个绘图区域分为numRows(行) * numCols(列)个子区域，按照从左到右，
    #从上到下的顺序依次给子区域进行编号，并将需要绘制的图画在编好的第plotNum(个)子区域中。
    #如果numRows，numCols，plotNum这三个数都小于10的话，可以把它们缩写成一个三位证书，例如subplot(223)和subplot(2, 2, 3)是相同的。
    #plt.subplot(221)表示分为两行两列，占用第一个，即第一行第一列的子图
    #最后传入的字典是x、y轴上的数值
    createPlot.ax1 = plt.subplot(111, frameon=False,**axprops)
    #树的宽度
    plotTree.totalW=float(getNumLeafs(inTree))
    #树的高度
    plotTree.totalD=float(getTreeDepth(inTree))
    #使箭头垂直，图形偏移
    plotTree.xOff=-0.5/plotTree.totalW
    plotTree.yOff=+1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()


createPlot(dataTest[1])