# -*- coding: utf-8 -*-
'''
获得数据集
n个样本, m个属性(不负责加一列1的处理)
X <x1, x2, x3, ...> (n, m)
Y <yk> (n,1)
'''
import numpy as np
import struct
import pandas as pd

'''
二类别高斯分布数据生成
features
X = <x1, x2>
classification
Y = <0, 1>
distribution
P(xi|Y=yk) ~ N(μik, Σi)
'''


def binary_label_data(N=100, prop=0.4):
    # 特征均值
    mean = [[2, 3], [8, 4]]
    # 特征协方差矩阵
    cov = [[1, 0], [0, 0.2]]
    # 各类别数量
    n1, n2 = int(N * prop), int(N * (1 - prop))
    # 类别1
    X1 = np.random.multivariate_normal(mean[0], cov, n1)
    # 类别0
    X0 = np.random.multivariate_normal(mean[1], cov, n2)
    # 合并两类数据
    X = np.concatenate((X1, X0), axis=0)
    # 类别标签
    Y1 = np.ones((n1, 1))
    Y0 = np.zeros((n2, 1))
    # 合并两类标签
    Y = np.concatenate((Y1, Y0), axis=0)
    return X, Y

'''
多类别高斯分布数据生成
n个样本, m个特征, k个种类
X (n, m)
'''


def multi_label_data(N=90, k=3):
    # 特征均值
    mean = [[1, 5], [3, 3], [5, 1]]
    # 特征协方差矩阵
    cov = [[0.3, 0], [0, 1]]
    # 各类别数量
    n = int(N / k)
    # 类别1
    X1 = np.random.multivariate_normal(mean[0], cov, n)
    # 类别2
    X2 = np.random.multivariate_normal(mean[1], cov, n)
    # 类别3
    X3 = np.random.multivariate_normal(mean[2], cov, n)
    X = np.concatenate((X1, X2), axis=0)
    X = np.concatenate((X, X3), axis=0)
    # 类别标签
    Y1 = np.zeros((n, 1))
    Y2 = np.ones((n, 1))
    Y3 = np.ones((n, 1)) * 2
    Y = np.concatenate((Y1, Y2), axis=0)
    Y = np.concatenate((Y, Y3), axis=0)
    return X, Y

'''
从文件读取二类别数据
'''


def load_cryotherapy_data(filename="dataset/Cryotherapy.xlsx"):
    # 从文件读取
    df = pd.read_excel(filename)
    X = df.values[:, :-1]
    Y = df.values[:, -1:]
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y

'''
读取多类别数据集特征
加载图片特征
'''


def load_image_set(filename):
    print("load image set...", filename)
    binfile = open(filename, 'rb')
    buffers = binfile.read()
    head = struct.unpack_from('>IIII', buffers, 0)
    print("head,", head)
    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    # [60000]*28*28
    bits = imgNum * width * height
    # like '>47040000B'
    bitsString = '>' + str(bits) + 'B'
    imgs = struct.unpack_from(bitsString, buffers, offset)
    binfile.close()
    imgs = np.reshape(imgs, [imgNum, 1, width * height])
    print("load imgs finished")
    return imgs

'''
读取多类别数据集标签
加载标签
'''


def load_label_set(filename):
    print("load label set", filename)
    binfile = open(filename, 'rb')
    buffers = binfile.read()
    head = struct.unpack_from('>II', buffers, 0)
    print("head,", head)
    imgNum = head[1]
    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])
    print('load label finished')
    return labels
