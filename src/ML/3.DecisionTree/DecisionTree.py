"""
Created on 2018/6/10
@author: AlanYx
"""
import operator
from math import log, log2
from collections import Counter
import pickle

# todo 完成一个画图的文件编写
# from decisionTreePlot as dtPlot
from numpy.ma import copy


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
    # todo 这个香农熵是某个具体值还是一个列表
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

    # 切分数据集的方式2==============================================
    # retDataSet = [data[:index] + data[index +1:] for data in dataSet for i ,v in enumerate(data) if i == index and v == value]
    # 切分数据集的方式2==============================================END
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    选择切分数据集的最优特征
    :param dataSet: 需要切分的数据集
    :return: bestFeature:切分数据集的最优特征列
    """
    # 选择切分数据集的最优特征方式1==============================================
    # 求第一行有多少列的feature,最后一列是label
    numFeatures = len(dataSet[0]) - 1
    # label的信息熵
    baseEntropy = calcShannonEnt(dataSet)
    # 最优的信息增益值和最优的feature编号
    bestInfoGain, bestFeature = 0.0, -1

    # 遍历所有的特征
    for i in range(numFeatures):
        # 获取每一个实例的feature,组成list集合
        # for example in dataSet:
        #     featList = dataSet[i]
        featList = [example[i] for example in dataSet]
        # 获取去重后的集合:使用set对list去重
        uniqueVals = set(featList)
        # 创建一个临时的信息熵
        newEntropy = 0.0
        # 遍历某一列的value集合，计算该列的信息熵
        # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到对熵求和
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # gain[信息增益]:划分数据集前后的信息变化，获取信息熵最大的值
        # 信息增益是熵的减少或者数据无序度的减少。最后，比较所有特征中的信息增益，返回最好划分特征的索引值
        infoGain = baseEntropy - newEntropy
        # todo 查看打印结果
        print("infoGain=", infoGain, "bestEntropy=", i, baseEntropy, newEntropy)
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
    # 选择切分数据集的最优特征方式1==============================================END

    # 选择切分数据集的最优特征方式2==============================================
    # 计算初始香农熵
    # numFeatures = len(dataSet[0]) - 1
    # baseEntropy2 = calcShannonEnt(dataSet)
    # bestInfoGain2 = 0
    # bestFeature2 = -1
    # # 遍历每一个特征
    # for i in range(numFeatures):
    #     # 对当前特征进行统计
    #     featureCount = Counter(data[i] for data in dataSet)
    #     # 计算分割前后的香农熵
    #     # todo 理解一下这一个长串的公式是怎么求解的
    #     newEntropy2 = sum(feature[1] / float(len(dataSet)) * calcShannonEnt(splitDataSet(dataSet, i, feature[0])) \
    #                       for feature in featureCount.items())
    #     # 更新值
    #     infoGain = baseEntropy2 - newEntropy2
    #     print("No.{0} feature info gain is {1:.3f}".format(i, infoGain))
    #     if infoGain >bestInfoGain2:
    #         bestInfoGain2 = infoGain
    #         bestFeature2 = i
    # return bestFeature2
    # 选择切分数据集的最优特征方式2==============================================END


def majorityCnt(classList):
    """
    选择出现次数最多的一个结果
    :param classList: label列的集合
    :return: 出现次数最多的结果
    """
    # 方式1==============================================
    classCount = {}
    for vote in classList:
        if vote not in classList:
            classCount[vote] = 0
        classCount += 1
    # classCount倒叙排列得到一个字典，则第一个就是出现次数最多的结果
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # todo 查看打印结果
    # print("sortedClassCount", sortedClassCount)
    return sortedClassCount[0][0]
    # 方式1==============================================END

    # 方式2==============================================
    # todo 查阅这个函数方法说明
    # majorLabel = Counter(classList).most_common(1)[0]
    # return majorLabel
    # 方式2==============================================END


def createTree(dataSet, labels):
    """
    创建决策树
    :param dataSet: 待创建决策树待训练数据集
    :param labels: 训练数据集对应的标签
    :return: myTree:创建完成的决策树
    """
    classList = [example[-1] for example in dataSet]
    # classList = []
    # for example in dataSet:
    #     classList.extend(example[-1])
    # 如果数据集的最后一列(即标签)的第一个值出现的次数==整个集合的数量，则只有一个类别，直接返回结果
    # 第一个停止条件：所有类标签完全相同
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集只有一列，则最初出现标签次数最多的一类，作为结果返回
    # 第二个停止条件：使用完所有特征，仍然不能将数据集划分为仅包含唯一类别的分组
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优列，得到最优列对应的标签索引
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 取得标签的名字
    bestFeatLabel = labels[bestFeat]
    # 初始化myTree
    myTree = {bestFeatLabel: {}}
    # 注：标签列表是可变对象，在python函数中作为参数时是传址引用，能够被全局修改
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    del (labels[bestFeat])
    # 取出最优列，然后对其分支做分类
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签
        subLabels = labels[:]
        # 遍历当前选择的特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        # print("myTree", value, myTree)
    return myTree


def classify(inputTree, featLabels, testVec):
    """
    对新数据进行分类
    :param inputTree: 已经训练好对数据集
    :param featLabels: 特征标签对应的名称
    :param testVec: 测试输入数据
    :return: classLabel: 分类的结果值
    """
    # 获取tree的根节点对于的key值
    firstStr = list(inputTree.keys())[0]
    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    # todo 为什么这样做分类，如何理解
    # 判断根节点获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树做分类
    featIndex = featLabels.index(firstStr)
    # 测试数据，找到根节点对应的label位置，就知道从输入数据的第几位开始分类
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    # todo 查看打印结果
    # print("firstStr:", firstStr, "secondDict:", secondDict,"key", key, "valueOfFeat", valueOfFeat)

    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    """
    将训练好的决策树模型存储起来，使用pickle模块
    :param inputTree: 之前训练好的决策树模型
    :param filename: 要存储的名称
    :return: None
    """
    # 方式1==============================================
    fw = open(filename, "wb")
    # todo 查询dump的作用： pickle 模块将对象转化为文件保存在磁盘上，在需要的时候再读取并还原
    pickle.dump(inputTree, fw)
    fw.close()
    # 方式1==============================================END

    # 方式2==============================================
    with open(filename, "wb") as fw:
        pickle.dump(inputTree, fw)
    # 方式2==============================================END


def grabTree(filename):
    """
    将前面存储的决策树模型使用pickle模块还原出来
    :param filename: 之前存储的决策树的文件名
    :return: 模型
    """
    fr = open(filename, "rb")
    return pickle.load(fr)


def fishTest():
    """
    测试动物是不是鱼类，并将结果用matplotlib画出来
    :return: None
    """
    # 创建数据和结果标签
    myDat, labels = createDataSet()
    # print(myDat, labels)
    # 计算label分类标签的香农熵
    calcShannonEnt(myDat)

    # 求第0列为1/0的列的数据集（排除第0列）
    print("1:", splitDataSet(myDat, 0, 1))
    print("2:", splitDataSet(myDat, 0, 0))

    # 计算最好的信息增益的列
    print(chooseBestFeatureToSplit(myDat))

    # todo 查询copy.deepcopy()函数的作用
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)
    # [1,1]表示取的分支上的节点位置，对应的结果值
    print(classify(myTree, labels, [1, 1]))

    # todo 待完成画图文件的编写
    # 画图可视化展示
    # dtPlot.creatPlot(myTree)


def contactLensesTest():
    """
    预测隐形眼镜的测试代码，并将结果画出来
    :return: None
    """
    # 加载隐形眼镜的文本文件数据
    fr = open("/Users/yinxing/03workspace/self/MLAction/src/ML/3.DecisionTree/lenses.txt")
    # 解析数据，获得特征的数据
    lenses = [inst.strip().split("\t") for inst in fr.readlines()]
    # 得到数据对应的标签
    lensesLabels = ["age", "prescript", "astigmatic", "tearRate"]
    # 使用上面创建的决策树代码，构造预测隐形眼镜的决策树
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    # 画图可视化展示
    # dtPlot.creatPlot(lensesTree)


if __name__ == '__main__':
    fishTest()
    # contactLensesTest()
