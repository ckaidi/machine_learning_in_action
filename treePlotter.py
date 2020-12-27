import matplotlib.pyplot as plt

#定义文本框和箭头格式
decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_arge=dict(arrowstyle="<-")

#获取叶节点的数目
def getNumLeafs(myTree):
    numLeafs=0
    firstStr=myTree.keys()[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys()[0]:
        if(type(secondDict[key])._name_=='dict'):
            numLeafs+=getNumLeafs(secondDict[key])
        else: numLeafs+=1
    return numLeafs

#获取树的层数
def getTreeDepth(myTree):
    maxDepth=0
    firstStr=myTree.keys()[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if(type(secondDict[key])._name_=='dict'):
            thisDepth=1+getTreeDepth(secondDict[key])
        else: thisDepth=1
        if(thisDepth>maxDepth): maxDepth=thisDepth
    return maxDepth

#绘制带箭头的注解
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',
    va="center",ha="center",bbox=nodeType,arrowprops=arrow_arge)

def createPlot():
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1=plt.subplot(111,frameon=False)
    plotNode('a decison node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()

createPlot()