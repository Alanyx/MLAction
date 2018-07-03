"""
Created on 2018/6/22
@author: AlanYx
"""
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename):
    """
    对文件逐行解析，加载数据
    :param filename:
    :return: dataMat,labelMat
    """
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        # todo 查看这个文件数据每一行是不是3列
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    """
    随机选择一个整数 todo 是为了随机选择一个alpha求取参数b？
    :param i: 第一个alpha的下标
    :param m: 所有alpha的数量
    :return: j-返回一个不为0的整数，在0~m之间的整数值
    """
    j = i
    while j == i:
        # todo 查阅random.uniform()函数的作用
        j = int(random.uniform(0, m))
    return j


def cliAlpha(aj, H, L):
    # todo 这个最大和最小值是如何选择和确定的？
    """
    调整aj的值，使aj处于L<=aj<=H
    :param aj: 目标值
    :param H: 最大值
    :param L: 最小值
    :return: aj
    """
    if aj >H:
        aj = H
    if aj <L:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    smo算法简单实现
    :param dataMatIn: 输入数据集
    :param classLabels: 类别标签
    :param C: 松弛常量（常量值）允许部分数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于1.0这两个目标的权重。
            可以通过调节该参数达到不同的结果。
    :param toler: 容错率: 指在某个体系中能减小一些因素或选择对某个系统产生不稳定的概率
    :param maxIter: 退出前最大循环次数
    :return: b-模型的常量值;alphas-拉格朗日乘子
    """
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)

    # 初始化b和alphas(alphas类似于权重)
    b = 0
    alphas = mat(zeros(m, 1))

    # 无任何alphas改变的情况下遍历次数
    iter = 0
    while (iter < maxIter):
        # w = calcWs(alphas, dataMatIn, classLabels)
        # print("w:", w)

        # 记录alphas是否已经进行优化，每次循环时设为0，然后再遍历整个集合
        alphaPairsChanged = 0
        for i in range(m):
            print("alpha=", alphas)
            print("lableMat=", labelMat)
            print("multiply(alphas, labelMat)=", multiply(alphas, labelMat))
            # 我们预测的类别为:y = w^T x[i] + b;其中w = Σ(1-n) a[n]*label[n]*x[n]
            fxi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            # 预测结果与真实值对比，计算误差Ei
            Ei = fxi - float(labelMat[i])

            # 约束条件(KKT条件是解决最优化问题时用到的一种方法。此处的最优化问题通常指:对于给定的一个函数，求其在指定作用域上的全局最小值)
            # 0<=alphas[i]<=C，但是0和C都是边界值，我们无法优化，因此需要增加一个alphas,减少另一个alphas
            # 发生错误的概率:labelMat[i] * Ei,若大于toler,才需要进行优化。对于正负号，考虑取绝对值
            # todo 什么是KKT条件？如何直观理解
            """
            # 检验训练样本(xi, yi)是否满足KKT条件
            yi*f(i) >= 1 and alpha= 0(在边界之外)
            yi*f(i) == 1 and 0<alpha<C(在边界上)
            yi*f(i) <= 1 and alpha=C(在边界之间)
            """
            if ((labelMat[i] * Ei < -toler) and (alphas[i] <C)) or ((labelMat[i] * Ei > toler) and (alphas[i] >0)):
                # 如果满足优化的条件，随机选取一个非i的一个点，进行优化比较
                j = selectJrand(i, m)
                # 预测j的结果
                fxj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fxj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # L和H用于将alphas[j]调整到0-C之间。如果L==H，就不做任何的改变，直接执行continue语句
                # labelMat[i] != labelMat[j]表示异侧，则相减；否则为同侧，就相加
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C+ alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                # 如果相同，就没有办法优化了
                if L == H:
                    print("L == H")
                    continue

                # eta是alphas[j]的最优修改量，如果eta==0，需要退出for循环的当前迭代过程
                # 参考《统计学习方法》李航-P125~P128<序列最小最优化算法>
                eta = 2.0 * dataMatrix






























