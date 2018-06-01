"""
Created on 2018/5/31
@author: AlanYx
"""

# 导入科学计算包numpy和运算符模块operator
from numpy import *
import operator
from os import listdir
from collections import Counter

from numpy.core.defchararray import title


def createDataSet():
    """
    创建数据集和标签

    调用的方式：
    import KNN
    group, labels = KNN.createDataSet()
    :return:
    """
    group = array([1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classfy0(inX, dataSet, labels, k):
    # 1.距离计算
    dataSetSize = dataSet.shape[0]
    diffMat = title(inX, (dataSetSize, 1)) - dataSet
    # 取平方
    sqDiffMat = diffMat ** 2
    # 将矩阵的每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方
    distances = sqDistances ** 0.5




#     todo 待完成


def test1():
    """
    第一个测试实例
    :return:
    """
    group, labels = createDataSet()
    print(str(group))
    print(str(labels))
    print(classfy0([0.1, 0.1], group, labels))
