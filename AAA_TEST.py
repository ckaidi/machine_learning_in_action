import numpy as np
from numpy.core.fromnumeric import shape

dataMat=[[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9]]
dataMat=np.matrix(dataMat)
a=np.ones((shape(dataMat)[0],1))
print(a)
b=5
a[dataMat[:,1]<=b]=-12
print(dataMat[:,1])
print(a)