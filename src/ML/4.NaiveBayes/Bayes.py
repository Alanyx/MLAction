"""
Created on 2018/6/7
@author: AlanYx
"""
from numpy import *


# 案例1：屏蔽社区留言板中的侮辱性言论

def loadDataSet():
    """
    加载数据集
    :return: 单词列表postingList,所属类别classVec
    """
    # [0,0,1,1,1......]
    # todo 如果句子的长度中词汇个数不一样，是怎么解决的?
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1:侮辱性言论 0:正常言论
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    """
    获取所有单词的集合
    :param dataSet: 数据集
    :return: 所有单词的集合（不含重复元素的单词列表）
    """
    vocabSet = set([])
    for document in dataSet:
        # |:用于求两个集合的并集
        vocabSet = vocabSet | set(document)
        # todo 查看一下这个列表的形式
        # print(list(vocabSet))
    return list(vocabSet)


# todo 这个方法输出的列表有什么意义？这个输入数据集是指哪个？是指加载的数据集吗
def setOfWord2Vec(vocabList, inputSet):
    """
    遍历查看该单词是否出现，出现则将该单词置1
    :param vocabList: 所有单词集合列表
    :param inputSet: 输入数据集
    :return: 匹配列表[0,1,0,1...]，其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
    """
    # 创建一个与词汇表等长的向量，将所有元素置为0
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    训练数据原版
    :param trainMatrix: 文件单词矩阵  如[[1,0,1,1,1....],[],[]...]
    :param trainCategory: 文件对应的类别  如[0,1,1,0....]，列表长度等于单词矩阵数，其中的1代表对应的文件是侮辱性文件，0代表不是侮辱性矩阵
    :return:
    """
    # 文件数
    # todo 为什么len(trainMatrix)是文件数，文件在这里指什么
    numTrainDocs = len(trainMatrix)
    # 单词数
    numWords = len(trainMatrix[0])
    # trainCategory中所有1的个数，就是侮辱性文件，除以文件总数就是侮辱性文件出现的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 构造单词出现的次数列表
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)

    # 整个数据集单词的出现次数列表
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        # 遍历所有文件，如果是侮辱性文件，就计算此侮辱性文件中出现侮辱性单词的个数;否则计算非侮辱性文件中出现侮辱性单词的个数
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # todo 每个单词出现的占比有什么含义？
    # 类别1，即侮辱性文档的[P(F1|C1),P(F2|C1),P(F3|C1),P(F4|C1),P(F5|C1)....]列表
    # 即 在1类别下，每个单词出现次数的占比   如#[1,2,3,5]/90->[1/90,...]
    p1Vect = p1Num / p1Denom
    # 类别0，即正常文档的[P(F1|C0),P(F2|C0),P(F3|C0),P(F4|C0),P(F5|C0)....]列表
    # 即 在0类别下，每个单词出现次数的占比
    p0Vect = p0Num / p0Denom
    return p0Vect, p1Vect, pAbusive


def trainNB1(trainMatrix, trainCategory):
    """
    训练数据优化版本
    :param trainMatrix: 文件单词矩阵
    :param trainCategory: 文件对应的类别
    :return:
    """
    # 文件数
    numTrainDocs = len(trainMatrix)
    # 单词数
    numWords = len(trainMatrix[0])
    # 侮辱性文件出现的概率
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 构造单词出现的次数列表  p0Num:正常词汇的统计  p1Num:侮辱词汇的统计
    # 为避免单词列表中的任何一个单词为0，而造成最后的乘积为0，所以每个单词出现的次数初始化为1
    p0Num = ones(numWords)
    p1Num = ones(numWords)

    


























if __name__ == '__main__':
    loadDataSet()
