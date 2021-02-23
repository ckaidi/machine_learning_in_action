from numpy import *
from numpy.linalg.linalg import tensorsolve
from numpy.matrixlib.defmatrix import matrix

def loadDataSet(fileName):
    numFeat=len(open(fileName).readline().split('\t'))-1
    dataMat=[];labelMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#用于计算最佳拟合曲线
def standRegers(xArr,yArr):
    xMat=matrix(xArr);yMat=matrix(yArr).T
    xTx=xMat.T*xMat
    #判断是否可逆，det求矩阵行列式，行列式为0则不可逆
    if linalg.det(xTx)==0.0:
        print ("This matrix is singular,cannot do inverse")
        return
    ws=xTx.I*(xMat.T*yMat)
    return ws

#局部加权线性回归函数
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat=matrix(xArr);yMat=matrix(yArr).T
    m=shape(xMat)[0]
    #创建单元矩阵
    weights=matrix(eye(m))
    for j in range(m):
        #权重值大小以指数级衰减
        diffMat=testPoint-xMat[j,:]
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx=xMat.T*(weights*xMat)
    if linalg.det(xTx)==0:
        print("This matrix is singular,cannot do inverse")
        return
    ws=xTx.T*(xMat.T*(weights*yMat))
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=shape(testArr)[0]
    yHat=zeros(m)
    for i in range(m):
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat

def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

dataMat,labelMat=loadDataSet('D:/毕业设计/DATA/43f/43features_18.txt')
#dataMat,labelMat=loadDataSet('DATA/Ch08/abalone.txt')
#yHat01=lwlrTest(dataMat[0:1999],dataMat[0:1999],labelMat[0:1999],0.1)
#yHat1=lwlrTest(dataMat[0:1999],dataMat[0:1999],labelMat[0:1999],1)
#yHat10=lwlrTest(dataMat[0:1999],dataMat[0:1999],labelMat[0:1999],10)
#print(rssError(labelMat[0:1999],yHat01.T))
#print(rssError(labelMat[0:1999],yHat1.T))
#print(rssError(labelMat[0:1999],yHat10.T))
ws=standRegers(dataMat,labelMat)
s=shape(ws)
ws=ws.tolist()
fr=open("ws_18.txt","w+")
for i in ws:
    fr.write(str(i[0]))
    fr.write("\t")
fr.close()
print(s)
print(ws)