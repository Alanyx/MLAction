"""
Created on 2018/6/20
@author: AlanYx
"""
from numpy import *
import matplotlib.pylab as plt
# todo for time import sleep 作用是什么？
from time import sleep
# todo import bs4 作用是什么？
import bs4
from bs4 import BeautifulSoup
import json
import urllib.request

def loadDataSet(filename):
    """
    加载数据
    解析以tab键分隔的文件中的浮点数
    :param filename:
    :return: dataMat-数据集，labelMat-类别标签
    """
    # 获取样本特征总数(不含最后一列标签)
    numFeat = len(open(filename).readline().split("\t")) - 1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        # 读取每一行
        lineArr = []
        # 删除每一行中以tab分隔的数据前后的空白符号
        currentLine = line.strip().split("\t")
        for i in range(numFeat):
            # 将数据添加到lineArr列表中，每一行数据组成一个行向量
            lineArr.append(float(currentLine[i]))
        # 将测试数据的输入数据部分存储到dataMat列表中
        dataMat.append(lineArr)
        # 将每一行的最后一个数据（即类别）存储到labelMat中
        labelMat.append(float(currentLine[-1]))
    return dataMat, labelMat

def standRegression(xArr, yArr):
    """
    线性回归
    :param xArr: 输入的样本数据（包含每个样本数据的特征）
    :param yArr: 输入的类别标签（包含每个样本的目标变量）
    :return: w-回归系数
    """
    # mat()函数:将数组转换为矩阵； mat().T:对矩阵进行转置
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    # 要用到xTx的逆矩阵，故先判断xTx是否可逆，条件是矩阵的行列式不为0
    # linalg.det()函数是用来求得矩阵的行列式：若行列式为0，则矩阵不为逆
    if linalg.det(xTx) == 0.0:
        print("this matrix is singular, cannot do inverse")
        return
    # 最小二乘法（根据公式，求得w的最优解）
    # xTx.I:矩阵求逆
    w = xTx.I * (xMat.T * yMat)
    return w


def regression1():
    """
    测试standRegression
    :return:
    """
    xArr, yArr = loadDataSet("/Users/yinxing/03workspace/self/MLAction/src/ML/8.Regression/data.txt")
    xMat = mat(xArr)
    yMat = mat(yArr)
    w = standRegression(xArr, yArr)
    fig = plt.figure()
    # add_subplot(349)函数的参数的意思:将画布分成3行4列图像画在从左到右从上到下第9块
    ax = fig.add_subplot(111)
    # scatter 的x是xMat中的第二列，y是yMat的第一列
    ax.scatter([xMat[:, 1].flatten()], [yMat.T[:, 0].flatten().A[0]])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * w
    ax.plot(xCopy[:, 1], yHat)
    plt.show()


# todo 待完成

if __name__ == '__main__':
    regression1()

















