"""
Created on 2018/5/31
@author: AlanYx
"""

# 导入科学计算包numpy和运算符模块operator
from numpy import array, tile, zeros, shape
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
    """
    分类
    :param inX: 输入向量
    :param dataSet: 输入对训练数据集
    :param labels: 标签向量
    :param k: 选择最近邻居的数量
    :return:
    """
    # 1.距离计算 此处采用欧式距离
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
    # todo 返回索引是什么作用
    sortedDisIndicies = distances.argsort()

    # 2.选择距离最小的k个点
    classCount = {}
    for i in range(k):
        # 找到该样本的类型
        # todo 这个labels[sortedDisIndicies[i]]返回的是什么类型为什么是"A","B"，没有明白
        voteIlbale = labels[sortedDisIndicies[i]]
        classCount[voteIlbale] = classCount.get(voteIlbale, 0) + 1

    # 3.排序并返回出现最多的那个类型  items()方法见备注
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


# ====================================================================================
def file2matrix(filename):
    """
    导入训练数据
    :param filename:数据文件路径
    :return: 数据矩阵returnMat和对应的类别classLabelVector
    """
    fr = open(filename)
    # 获得文件中的数据的行数
    numberOfLines = len(fr.readlines())
    # 生成一个空矩阵用于返回数据(3:代表3个特征)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        # todo 查一下这个函数的说明和打印一下结果
        # str.strip([char]) --返回移除字符串头尾指定的字符生成的新字符串
        line = line.strip()
        listFromLine = line.split("\t")
        # 每列的属性数据
        # returnMat[index, :]返回从index起始的所有列的数组
        # todo 查看一下类型是数组还是矩阵
        returnMat[index, :] = listFromLine[0:3]
        # 每列的类别数据：即label标签数据
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
        return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    归一化特征值：消除特征之间量级不同导致的影响
    :param dataSet: 数据集
    :return: 归一化后的数据集normDataSet,范围ranges,最小值minVals
    """
    # 计算属性的最大/最小值/范围
    # todo 为什么要取0？无参数不行吗，断点无参数看看结果
    minVals0 = dataSet.min()
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals

    # 归一化实现方式1：---------------------------
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 生成与最小值之差组成的矩阵
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 将最小值之差除以范围组成的矩阵
    normDataSet = normDataSet / tile(ranges, (m, 1))
    # 归一化实现方式1---end----------------------

    # 归一化实现方式2：---------------------------
    # normDataSet2 = (dataSet - minVals)/ranges
    # 归一化实现方式2---end----------------------
    return normDataSet, ranges, minVals


def datingClassTest():
    """
    约会网站的测试方法
    :return: 错误数errorCount
    """
    # 设置测试数据的一个比例（训练数据比例=1-hoRatio）
    hoRatio = 0.1
    # 从文件中加载数据
    # todo 文件加载路径尝试改变绝对路径和相对路径
    datingDataMat, datingLabels = file2matrix("/Users/yinxing/03workspace/self/MLAction/src/ML/2.KNN/datingTestSet2.txt")
    # 归一化数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # m为矩阵的第一维（行数）
    m = normMat.shape[1]
    # 设置测试的样本数量
    numTestVecs = int(m * hoRatio)
    # print("numTestVecs=",numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 对数据进行测试
        classifierResult = classfy0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is :%f" % (errorCount / float(numTestVecs)))
    print(errorCount)


def img2Vector(filename):
    """
    将图像数据转换为向量
    :param filename: 图片文件（此例中输入图片格式是 32 * 32）
    :return: 一维矩阵
    该函数将图像转换为向量：该函数创建 1 * 1024 的NumPy数组，然后打开给定的文件，
    循环读出文件的前32行，并将每行的头32个字符值存储在NumPy数组中，最后返回数组。
    """
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritringClassTest():
    # 1.导入数据
    hwLables = []
    # todo 文件加载路径尝试改变绝对路径和相对路径
    trainingFileList = listdir("2.KNN/trainingDigits")
    # todo 为什么取list的长度，直观理解是什么
    m = len(trainingFileList)
    trainingMat = zeros(m, 1024)
    # hwLabels存储0～9对应的index位置，trainingMat存放的每个位置对应的图片向量
    for i in range(m):
        fileNamestr = trainingFileList[i]
        # todo 下面两行[0]是什么操作？
        fileStr = fileNamestr.split(".")[0]
        classNumStr = int(fileStr.split("_"))[0]
        hwLables.append(classNumStr)
        # 将 32 * 32的矩阵转化为 1 * 1024矩阵
        trainingMat[i, :] = img2Vector("2.KNN/trainingDigits%s" % fileNamestr)

    # 2.导入测试数据
    testFileList = listdir("2.KNN/testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNamestr = testFileList[i]
        fileStr = fileNamestr.split(".")[0]
        classNumStr = int(fileStr.split("_"))[0]
        vectorUnderTest = img2Vector("2.KNN/testDigits/%s" % fileNamestr)
        classifierResult = classfy0(vectorUnderTest, trainingMat, hwLables, 3)
        print("the classifier came back with: %d, the real answer is:%d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\n the total number of error is:%d" % errorCount)
    print("\n the total error rate is: %" % (errorCount / float(mTest)))


if __name__ == '__main__':
    test1()
    # datingClassTest()
    # handwritringClassTest()
