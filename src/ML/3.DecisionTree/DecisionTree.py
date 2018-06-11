"""
Created on 2018/6/10
@author: AlanYx
"""
import operator
from math import log, log2
from collections import Counter


# todo 完成一个画图的文件编写
# from decisionTreePlot as dtPlot

def createDataSet():
    """
    创建数据集
    :return: 返回数据集和label标签
    """
    # dataSet前两列是特征，最后一列是每条对应的分类标签
    dataSet = [[1, 1, "yes"],
               [1, 1, "yes"],
               [1, 0, "no"],
               [0, 1, "no"],
               [0, 1, "no"]]
    # label:露出水面/脚蹼
    # 注意：这里labels是dataSet中特征的具体含义，并不是对应的分类标签或者目前变量
    labels = ["no surfacing", "flippers"]
    # 返回
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    计算给定数据集的香农熵
    推荐结合李航《统计学习方法》信息增益算法5.1理解
    :param dataSet: 数据集
    :return: shannonEnt:返回每一组特征中的某个分类下，香农熵的信息期望
    """
    # 香农熵计算方式1==============================
    # 求list的长度：即训练数据集的样本个数
    numEntries = len(dataSet)
    # 计算不同类别label出现的样本个数（字典形式）
    labelCounts = {}

    # 唯一元素和其出现的次数
    for featVec in dataSet:
        # 存储当前实例的类别：即每行数据的最后一个数据
        currentLabel = featVec[-1]
        # 为所有类创建字典：若当前的键值不存在，则扩展将当前键值加入字典；若存在，则加1
        # 每一对键值记录类当前类别出现的次数
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
        # todo 打印出来查看一下结果
        # print("featVec:"+featVec , +"labelCount:"+labelCount)

    # 对label标签的占比，求出label标签的香农熵
    shannonEnt = 0.0
    # 计算labelCount个类的经验熵
    for key in labelCounts:
        # 使用各个类标签的发生频率计算各类别出现的概率
        prob = float(labelCounts[key]) / numEntries
        # 计算香农熵，以2为底求对数
        shannonEnt -= prob * log2(prob)
        # shannonEnt -= prob * log(prob, 2) 效果一样
        # todo 打印出来查看一下结果
        # print("prob:" + prob, "prob * log2(prob):" + prob * log2(prob), "shannonEnt:" + shannonEnt)
    # 计算方式1==============================END


    # 香农熵计算方式2==============================
    # 统计标签出现的次数
    labelCounts2 = Counter(data[-1] for data in dataSet)
    # 计算概率
    prob2 = [p[1] / len(dataSet) for p in labelCounts2.items()]
    # 计算香农熵
    shannonEnt = sum([-p * log2(p) for p in prob2])
    # 计算方式2==============================END
    return shannonEnt

def splitDataSet(dataSet, index, value):
    """
    划分数据集 ：通过遍历dataSet数据集，求出index对应的column列值为value的行
    即依据index列进行分类，若index列的数据等于value时，就要将index划分到创建的新数据集中
    :param dataSet: 待划分数据集
    :param index: 每一行的index列（划分的特征）
    :param value: index列对应的value值（需要返回的特征的值）
    :return: index列为value的数据集（该数据需排除index列）
    """
    # 切分数据集的方式1==============================================
    retDataSet = []
    for featVec in dataSet:
        # index列为value的数据集（该数据集需要排除的index列）
        # 判断index列的值是否为value
        if featVec[index] == value:
            # 切分index用来划分， [:index]表示取featVec的前index行
            reducedFeatVec = featVec[:index]
            # [index + 1:]表示跳过index的index+1行，取接下来的数据
            reducedFeatVec.extend(featVec[index + 1:])
            # 收集结果值：index列为value的行（该行需要排除index列）
            retDataSet.append(reducedFeatVec)
    # 切分数据集的方式1==============================================END





























if __name__ == '__main__':
    createDataSet()
