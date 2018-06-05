"""
Created on 2018/5/31
@author: AlanYx
"""

# 导入科学计算包numpy和运算符模块operator
from numpy import array,tile
import operator
from os import listdir
from collections import Counter
import urllib

from numpy.core.defchararray import title


def createDataSet():
    """
    创建数据集和标签

    调用的方式：
    import KNN
    group, labels = KNN.createDataSet()
    :return:
    """
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classfy0(inX, dataSet, labels, k):
    # 1.距离计算
    # shape[0]是第二维的长度（即列数）；shape[1]是第一维的长度（即行数）
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    #  todo 看一下diffMat长啥样
    # print(diffMat)
    # 取平方
    sqDiffMat = diffMat ** 2
    # 将矩阵的每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方
    distances = sqDistances ** 0.5
    # argsort()根据距离从小到大排序，然后返回对应的索引
    sortedDisIndicies = distances.argsort()

    # 2.选择距离最小的k个点
    classCount = {}
    for i in range(k):
        # 找到该样本的类型
        voteIlbale = labels[sortedDisIndicies[i]]
        classCount[voteIlbale] = classCount.get(voteIlbale, 0) + 1

    # 3.排序并返回出现最多的那个类型
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def test1():
    """
    第一个测试实例
    :return:
    """
    group, labels = createDataSet()
    print(str(group))
    print(str(labels))
    print(classfy0([1.1, 1.2], group, labels, 3))


if __name__ == '__main__':
    test1()
