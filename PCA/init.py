# -*- coding: utf-8 -*-
import numpy as np
from PCA import *
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import math
import scipy.stats as st

'''
一个类别
    生成三维数据
    第三维方差远小于其他两维
    画出主成分
两个类别
    数据旋转
    降维后类别
'''

def generate_data(N=50, K=3):
    # Mean
    mean = [1, 1, 1]
    # # 协方差矩阵
    cov = [[10, 0, 0], [0, 10, 0], [0, 0, 0.001]]
    # 生成数据
    X = np.random.multivariate_normal(mean, cov, N)
    Y = np.zeros(N, dtype=np.int8)
    # 如果生成多类别数据则执行该部分
    for i in range(K - 1):
        mean = [i + 2, i + 2, i + 2]
        X0 = np.random.multivariate_normal(mean, cov, N)
        X = np.concatenate((X, X0), axis=0)
        # 记录类别
        Y0 = np.ones(N, dtype=np.int8) + i
        Y = np.concatenate((Y, Y0))
    return X, Y

'''
绕非方差最小的轴旋转 theta 度
'''

def data_rotate(X):
    # 绕x轴旋转60度
    d = -math.pi * 60 / 18.0
    # 其他轴sin值为0, cos值为1
    # 旋转矩阵
    Mx = np.asarray([
        [1, 0, 0],
        [0, math.cos(d), -math.sin(d)],
        [0, math.sin(d), math.cos(d)]
    ])
    My = np.asarray([
        [math.cos(d), 0, math.sin(d)],
        [0, 1, 0],
        [-math.sin(d), 0, math.cos(d)]
    ])

    Mz = np.asarray([
        [math.cos(d), -math.sin(d), 0],
        [math.sin(d), math.cos(d), 0],
        [0, 0, 1]
    ])
    # 旋转
    return np.dot(X, Mx)

'''
数据三维空间
'''

def show_3D(data, data_c, title='3D'):
    plt.figure(title)
    # 获得三个维度
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    # 创建三维子图
    ax = plt.subplot(111, projection='3d')
    ax.set_title(title)
    # 三维散点
    ax.scatter(x, y, z, c=data_c)
    plt.savefig('dataset/' + title + '.png')

'''
人工生成3维数据, 降维为2维
'''

def trans_3_to_2():
    # 生成数据
    data, data_c = generate_data(50, 1)
    # 旋转数据
    datax = data_rotate(data)
    # 展示旋转前后的数据
    # show_3D(data, data_c, 'before rotate')
    # show_3D(datax, data_c, 'after rotate')
    # 提取主成分, 得到二维数据
    # plt.figure()
    X, Y = PCA(datax, 2)
    # show_3D(Y, data_c, 'After PCA')

    print('旋转前', [np.cov(data[:, d]) for d in range(data.shape[1])])
    print('旋转后', [np.cov(datax[:, d]) for d in range(datax.shape[1])])
    print('主成分', [np.cov(X[:, d]) for d in range(X.shape[1])])

    # plt.figure('Principal components')
    # plt.title('Principal components')
    # # print(np.concatenate((data, datax, X), axis=1))
    # plt.scatter(X[:, 0], X[:, 1], c=data_c)
    # plt.savefig('dataset/PCA.png')
    # plt.show()

'''
MNIST手写数据集重建
'''

def construct_mnist():
    # 主要成分
    K = 1
    # 手写数字
    num = 9
    # 样本数量
    N = 100
    print('read from MNIST_test.txt...')
    data = np.loadtxt('dataset/MNIST_test.txt', delimiter=',')
    # 切分 标签和 特征
    Y = data[:, 0]
    X = data[:, 1:]
    ######单一数字######
    # 获得某个手写数字的所有下标
    indices = np.argwhere(Y == num)
    # 获得所有该数字的样本
    X_n = X[indices][:N]
    # 展示原始图片
    slice_imgs(X_n, 'original')

    # 主成分分析 特征重建
    X_n_k, re_X_n = PCA(np.asarray(X_n).reshape((N, 784)), K)

    # 展示重建图片
    slice_imgs(np.real(re_X_n), 'reconstruct')

    # 每张图片的信噪比
    print('SNR of each picture...')
    print([compute_SNR(X_n[i], re_X_n[i]) for i in range(N)])

'''
展示图片
'''

def slice_imgs(X, filename='default', size=28):
    N, M = X.shape[0], X.shape[1]
    # 每行每列的个数
    nums = int(np.sqrt(N))
    # 主面板
    main_img = Image.new('L', (size * nums, size * nums))
    # 向主面板添加图片
    for i in range(N):
        # 数组转图片
        img = Image.fromarray(np.asarray(X[i]).reshape(size, size))
        # 计算图片位置
        loc = (int(i / nums) * size, (i % nums) * size)
        # 放置图片
        main_img.paste(img, loc)
    # Image.show()
    main_img.show()
    main_img.save('dataset/' + filename + '_mnist.png')

'''
计算原图片和重建图片的信噪比
'''

def compute_SNR(X, re_X):
    # 原始图片
    X = np.asarray(X).reshape((28, 28))
    # 重建图片
    re_X = np.asarray(re_X).reshape((28, 28))
    # 信噪比
    R = math.pow(np.linalg.norm(X, 2) / np.linalg.norm(X - re_X, 2), 2)
    # 转换单位
    return '{:.2f}'.format(10 * math.log(R, 10))

if __name__ == '__main__':
    # r人工生成数据
    #trans_3_to_2()
    # mnist降维和重建
    print('PCA on mnist and reconstruct...')
    construct_mnist()
