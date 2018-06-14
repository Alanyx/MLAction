"""
Created on 2018/6/5
@author: AlanYx
"""

from numpy import array, mat, shape, ones, exp, random, arange
# from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    """
    加载并解析数据
    :param fileName:要解析的文件所在的位置
    :return:dataMat--原始数据的特征; labelMat--原始数据的标签
    """
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split()
        # 为了方便计算，将x0的值设为1.0,即在每一行的开头添加一个1.0作为x0
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    """
    sigmoid跳跃函数
    :param inX:
    :return: 1.0/(1+ exp(-inx))
    """
    # Tanh是Sigmoid的变形，与 sigmoid 不同的是，tanh 是0均值的。因此，实际应用中，tanh 会比 sigmoid 更好。
    # todo 了解一下Tanh函数，均值是如何计算的
    return 2 * 1.0 / (1 + exp(-2 * inX)) - 1


def gradAscent(dataMatIn, classLabels):
    """
    正常的梯度上升法
    :param dataMatIn: 输入数据的特征List
    :param classLabels: 输入数据的类别标签
    :return: 最佳回归系数
    """
    # 数组转换为矩阵  例: [[1,1,2],[1,1,3]...]
    dataMatrix = mat(dataMatIn)
    # classLabels 是类别标签，它是一个 1*100 的行向量。为了便于矩阵计算，需要将该行向量转换为列向量，做法是将原向量转置，再将它赋值给labelMat
    # 矩阵的转置: transpose()转置函数
    labelMat = mat(classLabels).transpose()

    # m:数据数量; n:样本数量（特征数量）
    m, n = shape(dataMatrix)
    # alpha:向目标移动的步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    # weights代表回归系数， 此处ones((n, 1))创建一个长度和特征数相同的矩阵，其中的数全都是1
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        # labelMat是真实值
        error = labelMat - h
        # 0.001* (3*m)*(m*1) 表示在每一个列上的一个误差情况，最后得出 x1,x2,xn的系数的偏移量
        # todo 查一下报错的原因
        weights = weights + alpha * dataMatrix.transpose() * error
    return array(weights)


def stocGradAscent0(dataMatrix, classLabels):
    """
    随机梯度下降：
    a.一般的梯度下降算法在每次更新数据集时都需要遍历整个数据集，计算复杂度比较高
    b.随机梯度下降一次只用一个样本点来更新回归系数
    :param dataMatrix: 输入数据的数据特征矩阵（除去最后一列）
    :param classLabels: 输入数据的类别标签（最后一列）
    :return: 最佳回归系数weights
    """
    m, n = shape(dataMatrix)
    alpha = 0.01
    # n * 1的元素全为1的数组
    weights = ones(n)
    for i in range(m):
        # todo 小结sum函数的用法
        # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn,此处求出的 h 是一个具体的数值，而不是一个矩阵
        h = sigmoid(sum(dataMatrix[i] * weights))
        # 计算真实值与预测值之间的差值，然后按照该差值调整回归系数
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return array(weights)


def stoGradAscent1(dataMatrix, classLabels, numIter=150):
    """
    改进版随机梯度下降，使用随机的一个样本来更新回归系数
    :param dataMatrix: 输入数据的数据特征矩阵（除去最后一列）
    :param classLabels: 输入数据的类别标签（最后一列）
    :param numIter: 迭代次数
    :return: 最佳回归系数weights
    """
    # m:数据数量; n:样本数量（特征数量）
    m, n = shape(dataMatrix)
    # 创建与列数相同的矩阵的系数矩阵
    weights = ones(n)
    # 随机梯度，循环150次，观察是否收敛
    for j in range(numIter):
        # [0,1,2,3...,m-1]
        # 这里必须要用list，不然后面的del没法使用
        dataIndex = list(range(m))
        for i in range(m):
            # 随着i和j的不断增大，alpha会随着迭代不断减小，后面有一个常数，永远不为0
            alpha = 4 / (1.0 + j + i) + 0.0001
            # random.uniform(x, y) 方法将随机生成下一个实数，它在[x,y]范围内,x是这个范围内的最小值，y是这个范围内的最大值。
            # todo 小结这个uniform()用法
            randIndex = int(random.uniform(0, len(dataIndex)))
            # sum(dataMatrix[i] * weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+zn*xn   h:预测值
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]] * weights))
            error = classLabels[dataIndex[randIndex]] - h
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]
            del (dataIndex[randIndex])
    return weights


def plotBestFit(dataArr, labelMat, weights):
    """
    数据可视化展示
    :param dataArr: 样本数据的特征
    :param labelMat: 样本数据的标签
    :param weights: 回归系数
    :return: None
    """
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            # todo append后面的矩阵点是什么含义？
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c="red", marker="s")
    ax.scatter(xcord2, ycord2, s=30, c="green")
    x = arange(-3.0, 3.0, 0.1)
    """
    y的计算：理论上是这个样子
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    w0*x0+w1*x1+w2*x2=f(x)
    x0最开始就设置为1叻， x2就是我们画图的y值，而f(x)被我们磨合误差给算到w0,w1,w2身上去了
    所以： w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2
    """
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def simpleTest():
    """
    测试方法
    :return:
    """
    # 加载数据
    dataMat, labelMat = loadDataSet("/Users/yinxing/03workspace/self/MLAction/src/ML/5.Logistic/TestSet.txt")
    # 训练模型: f(x)=a1*x1+b2*x2+..+nn*xn中 (a1,b2, .., nn).T的矩阵值
    # 由于数组没有复制N份，array的乘法就是乘法
    dataArr = array(dataMat)
    # todo 尝试3种随机梯度下降，观察效果
    # weights = gradAscent(dataArr, labelMat)
    # weights = stoGradAscent0(dataArr, labelMat)
    weights = stoGradAscent1(dataArr, labelMat)
    # print(weights)
    # 数据可视化
    plotBestFit(dataArr, labelMat, weights)


# =======================================================================
def classifyVector(inX, weights):
    """
    分类函数：根据回归系数和特征向量来计算Sigmoid的值，大于0.5函数返回1，否则返回0
    :param inX: 特征向量
    :param weights: 梯度下降计算得到的回归系数
    :return: 如果pro>0.5,函数返回1；否则返回0
    """
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    """
    打开测试集和训练集，并对数据进行格式化处理
    :return: errorRate--分类错误率
    """
    frTrain = open("/Users/yinxing/03workspace/self/MLAction/src/ML/5.Logistic/horseColicTraining.txt")
    frTest = open("/Users/yinxing/03workspace/self/MLAction/src/ML/5.Logistic/horseColicTest.txt")
    trainingSet = []
    trainingLabels = []
    # 解析训练数据集中的数据特征和标签
    for line in frTrain.readlines():
        currLine = line.strip().split("\t")
        lineArr = []
        # todo 为什么是21？
        for i in range(21):
            lineArr.append(float(currLine[21]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 使用改进的梯度下降算法，求得最佳回归系数trainWeights
    trainWeights = stoGradAscent1(array(trainingSet), trainingLabels, 500)
    # trainWeights = stocGradAscent0(array(trainingSet), trainingLabels)

    errorCount = 0
    numTestVec = 0.0
    # 读取测试数据集进行测试，计算分类错误的样本条数和最终的错误率
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split("\t")
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1.0
        errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is :%f" % errorRate)
    return errorRate


def multiTest():
    """
    调用colicTest() 10次并求结果的平均值
    :return: None
    """
    numTests = 10
    erroeSum = 0.0
    for k in range(numTests):
        erroeSum += colicTest()
    print("after %d iterations the average error rate is %f" % (numTests, erroeSum / float(numTests)))


if __name__ == '__main__':
    # simpleTest()
    # colicTest()
    multiTest()