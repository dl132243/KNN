import numpy as np
import matplotlib.pyplot as pl


#导入数据集,并把数据特征和标签分别用数组表示
def loadFile(filename):
    f = open(filename)
    numFeat = len(f.readline().split('\t')) - 1  # 获取数据集的特征量，默认最后一列为标签
    lineArr = f.readlines()
    dataMat = []
    labelMat = []
    # 把数据特征和标签用数组表示
    for line in lineArr:
        curArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            curArr.append(float(curLine[i]))
        dataMat.append(curArr)
        labelMat.append(curLine[-1])
    dataArr = np.array(dataMat)
    labelArr = np.array(labelMat)
    return dataArr,labelArr

#数据归一化(KNN算法倾向于数据值大的特征)
def norm(dataSet):
    minVals = dataSet.min(0) #选取当前列中最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals

    m,n = np.shape(dataSet)
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet /np.tile(ranges,(m,1))
    return normDataSet


#KNN算法，计算未知数据到已经标签的距离并分类标签
def classify(dataMat,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(dataMat,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    indexdistance = distances.argsort() #对数组从小到大排序，返回索引。
    num_dict= {}
    for i in range(k):
        curLabels = labels[indexdistance[i]]
        num_dict[curLabels] = num_dict.get(curLabels,0) +1
        mostTimes = sorted(num_dict.items(), key=lambda x: x[1], reverse=True)[0][0]
    return mostTimes

