"""
Created on 2018/6/7
@author: AlanYx
"""
import feedparser
import numpy as np
import re
import random
from operator import itemgetter


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
            # 这个后面应该注释掉，因为对你没什么用，这只是为了辅助调试的
            print("the word: %s is not in my vocabulary!" % word)
            pass
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
    朴素贝叶斯分类修正版(为什么要修正查阅书籍) todo 补充原因
    :param trainMatrix: 文件单词矩阵
    :param trainCategory: 文件对应的类别
    :return:
    """
    # 文件数
    trainDocsNum = len(trainMatrix)
    # 单词数
    wordsNum = len(trainMatrix[0])
    # 侮辱性文件出现的概率
    # 因为侮辱性的被标记为了1， 所以只要把他们相加就可以得到侮辱性的有多少
    # 侮辱性文件的出现概率，即trainCategory中所有的1的个数，
    # 代表的就是多少个侮辱性文件，与文件的总数相除就得到了侮辱性文件的出现概率
    pAbusive = sum(trainCategory) / float(trainDocsNum)
    # 构造单词出现的次数列表  p0Num:正常词汇的统计  p1Num:侮辱词汇的统计
    # 原版，变成ones是修改版，这是为了防止数字过小溢出
    # p0num = np.zeros(words_num)
    # p1num = np.zeros(words_num)
    # 为避免单词列表中的任何一个单词为0，而造成最后的乘积为0，所以每个单词出现的次数初始化为1
    p0Num = ones(wordsNum)
    p1Num = ones(wordsNum)
    # 整个数据集单词出现的次数(原来是0，后面改为2)
    p0NumAll = 2.0
    p1NumAll = 2.0

    for i in range(trainDocsNum):
        # 遍历所有的文件，如果是侮辱性文件，就计算此文件中出现侮辱性单词的个数
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1NumAll += np.sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p1NumAll += np.sum(trainMatrix[i])
    # 后面改成取log函数
    p0Vec = np.log(p0Num / p0NumAll)
    p1Vec = np.log(p1Num / p1NumAll)
    return p0Vec, p1Vec, pAbusive


def classifyNaiveBayes(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    使用算法：
        将乘法转换为加法
        乘法：P(C|F1F2...Fn) = P(F1F2...Fn|C)P(C)/P(F1F2...Fn)
        加法：P(F1|C)*P(F2|C)....P(Fn|C)P(C) -> log(P(F1|C))+log(P(F2|C))+....+log(P(Fn|C))+log(P(C))
    :param vec2Classify: 待测数据[0,1,1,1,1...]，即要分类的向量
    :param p0Vec: 类别0，即正常文档的[log(P(F1|C0)),log(P(F2|C0)),log(P(F3|C0)),log(P(F4|C0)),log(P(F5|C0))....]列表
    :param p1Vec: 类别1，即侮辱性文档的[log(P(F1|C1)),log(P(F2|C1)),log(P(F3|C1)),log(P(F4|C1)),log(P(F5|C1))....]列表
    :param pClass1: 类别1，侮辱性文件的出现概率
    :return: 类别1或0
    """
    # 使用 NumPy 数组来计算两个向量相乘的结果，这里的相乘是指对应元素相乘，即先将两个向量中的第一个元素相乘，然后将第2个元素相乘，以此类推。
    # vec2Classify * p1Vec:将每个词与其对应的概率相关联起来
    # 可以理解为:1.单词在词汇表中的条件下，文件是good类别的概率 也可以理解为:2.在整个空间下，文件既在词汇表中又是good类别的概率
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def bagWords2Vec(vocabList, inputSet):
    """
    todo 对比两者的不同点
    词袋(和setOfWord2Vec作对比)
    :param vocabList: 所有单词集合列表
    :param inputSet: 输入数据集
    :return: 所有单词的集合（不含重复元素的单词列表）
    """
    result = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            result[vocabList.index(word)] += 1
        else:
            # todo 查阅.format()的作用
            print("the word: {} is not in my vocabulary".format(word))
    return result


def testingNB():
    """
    测试朴素贝叶斯算法
    :return: None
    """
    # 1.加载数据集
    listPost, listClasses = loadDataSet()
    # 2.创建单词集合
    vocabList = createVocabList(listPost)

    # 3.计算单词是否出现并创建数据矩阵
    trainMat = []
    for postIn in listPost:
        # 返回m*len(vocabList)的矩阵，记录的都是0,1信息,每一条记录的长度都和单词列表相同
        trainMat.append(setOfWord2Vec(vocabList, postIn))
    # 4.训练数据
    p0v, p1v, pAbusive = trainNB1(np.array(trainMat), np.array(listClasses))
    # 5.测试数据
    testOne = ["love", "my", "dalmation"]
    testOneDoc = np.array(setOfWord2Vec(vocabList, testOne))
    print("the result is:{}".format(classifyNaiveBayes(testOneDoc, p0v, p1v, pAbusive)))
    testTwo = ["stupid", "garbage"]
    testTwoDoc = np.array(setOfWord2Vec(vocabList, testTwo))
    print("the result is:{}".format(classifyNaiveBayes(testTwoDoc, p0v, p1v, pAbusive)))


# ===================项目案例2: 使用朴素贝叶斯过滤垃圾邮件===================
def textParse(bigStr):
    """
    词划分
    :param bigStr: 某个被拼接后的字符串
    :return: 全部小写的词列表，去掉少于2个字符的字符串
    """
    # 推荐使用\W+代替\W*
    # 因为\W*会匹配empty patten,在py3.5之后就会出现问题（todo 尝试一下），对re.split的影响
    tokenList = re.split(r"\W+", bigStr)
    if len(tokenList) == 0:
        print(tokenList)
    return [tok.lower() for tok in tokenList if len(tok) > 2]


def spamTest():
    """
    对贝叶斯垃圾邮件分类起进行自动化处理
    :return: None
    """
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        # 添加垃圾邮件信息
        # 这里需要做一个说明，为什么我会使用try except 来做
        # 因为我们其中有几个文件的编码格式是 windows 1252　（spam: 17.txt, ham: 6.txt...)
        # 这里其实还可以 :
        # import os
        # 然后检查 os.system(' file {}.txt'.format(i))，看一下返回的是什么
        # 如果正常能读返回的都是：　ASCII text
        # 对于except需要处理的都是返回： Non-ISO extended-ASCII text, with very long lines
        try:
            # todo 完善文件路径
            words = textParse(open("").format(i).read())
        except:
            # todo 完善文件路径
            words = textParse(open("".format(i), encoding="Windows 1252").read())
        docList.append(words)
        fullText.extend(words)
        classList.append(1)
        try:
            # 添加非垃圾邮件
            # todo 完善文件路径
            words = textparse(open('../../../input/4.NaiveBayes/email/ham/{}.txt'.format(i)).read())
        except:
            # todo 完善文件路径
            words = textParse(
                open('../../../input/4.NaiveBayes/email/ham/{}.txt'.format(i), encoding='Windows 1252').read())
        docList.append(words)
        fullText.extend(words)
        classList.append(0)
    # 创建词汇表
    vocabList = createVocabList(docList)

    # 生成10个随机数，为避免警告将每个数都转换为整型
    testSet = [int(num) for num in random.sample(range(50), 10)]
    # 在原来的trainingSet中去掉这10个数
    trainingSet = list(set(range(50) - set(testSet)))

    trainingMat = []
    trainingClass = []
    for docIndex in trainingSet:
        trainingMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        trainingClass.append(classList[docIndex])
    p0v, p1v, pSpam = trainNB1(np.array(trainingMat), np.array(trainingClass))

    # 开始测试
    errorCount = 0
    for docIndex in testSet:
        wordVec = setOfWord2Vec(vocabList, docList[docIndex])
        if classifyNaiveBayes(np.array(wordVec), p0v, p1v, pSpam) != classList[docList]:
            errorCount += 1
    print("the error rate is:{}".format(errorCount / len(testSet)))


# ===================项目案例3: 使用朴素贝叶斯从个人广告中获取区域倾向===================
def calcMostFreq(vocabList, fullText):
    """
    Rss源分类器及高频词去除函数
    :param vocabList:
    :param fullText:
    :return:
    """
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    # todo 查询itemgetter作用
    sortedFreq = sorted(freqDict.items(), key=itemgetter(1), reverse=True)
    # 返回前30个高频词
    return sortedFreq[0:30]


def localWords(feed1, feed0):
    """
    与spamTest()函数基本相同，参照理解
    :param feed1:
    :param feed0:
    :return:
    """
    docList = []
    classList = []
    fullText = []
    # 找出两个中最小的一个
    minLen = min(len(feed0), len(feed1))
    for i in range(minLen):
        # 类别1
        # todo feed1["entries"][i]["summary"]这个参数的含义是什么
        wordList = textParse(feed1["entries"][i]["summary"])
        # todo appned和extend的区别在哪？
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        # 类别0
        wordList = textParse(feed0["entries"][i]["summary"])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    # 删除高频词（高频词很大概率上是没有意义的词汇，删除对结果影响小）
    top30Words = calcMostFreq(vocabList, fullText)
    for pair in top30Words:
        # todo 为什么是pair[0]
        if pair[0] in vocabList:
            vocabList.remove(pair[0])

    # 获取训练数据和测试数据
    # todo random.sample(range(2 * minLen), 20)含义是什么
    # 随机生成10个数，为了避免警告每个数强转为整型
    testSet = [int(num) for num in random.sample(range(2 * minLen), 20)]
    # 在原来的trainingSet中去掉这10个数
    trainingSet = list(set(range(2 * minLen)) - set(testSet))

    # 把这些训练集和测试集变成向量的形式
    trainingMat = []
    trainingClass = []
    for docIndex in trainingSet:
        trainingMat.append(bagWords2Vec(vocabList, docList[docIndex]))
        trainingClass.append(classList[docIndex])
    p0v, p1v, pspam = trainNB1(np.array(trainingMat), np.array(trainingClass))
    errorCount = 0
    for docIndex in testSet:
        wordVec = bagWords2Vec(vocabList, docList[docIndex])
        if classifyNaiveBayes(np.array(wordVec), p0v, p1v, pspam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is {}".format(errorCount / len(testSet)))
    return vocabList, p0v, p1v

def testRss():
    """

    :return:
    """
    ny = feedparser.parse("http://newyork.craigslist.org/stp/index.rss")
    sf = feedparser.parse("http://sfbay.craigslist.org/stp/index.rss")
    vocabList, pSf, pNy = localWords(ny, sf)
    # 返回值都没用上，可以用_, _, _代替



def getTopWords():
    """

    :return:
    """
    # todo 查阅feedparser.parseu作用
    ny = feedparser.parse("http://newyork.craigslist.org/stp/index.rss")
    sf = feedparser.parse("http://sfbay.craigslist.org/stp/index.rss")
    vocabList, pSf, pNy = localWords(ny, sf)
    topNy = []
    topSf = []
    for i in range(len(pSf)):
        # todo 为什么是-6.0？
        if pSf[i] > -6.0:
            topSf.append(vocabList[i], pSf[i])
        if pNy[i] > -6.0:
            topNy.append(vocabList[i], pNy[i])
    sortedSf = sorted(topSf, key=lambda pair: pair[1], reverse=True)
    sortedNy = sorted(topNy, key=lambda pair: pair[1], reverse=True)
    print("\n----------- this is SF ---------------\n")
    for item in sortedSf:
        print(item[0])
    print("\n----------- this is NY ---------------\n")
    for item in sortedNy:
        print(item[0])


if __name__ == '__main__':
    # testingNB()
    # spamTest()
    # testRss()
    getTopWords()