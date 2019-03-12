# -*- coding: utf-8 -*-
from dataset import *
from kmeans import *
from GMM import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

'''
展示聚类结果
'''

def show(X, Y, title = 'default'):
    plt.figure(title)
    plt.title('maybe need to run more times...')
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.show()

'''
Adjust Rand Index
'''

def ARI(Y, Y_):
    K, N = len(np.unique(Y)), len(Y)

    conting_matrix = np.zeros((K, K))
    for i in range(len(Y)):
        conting_matrix[int(Y[i])][int(Y_[i])] += 1
    sum_nij, sum_ai, sum_bj = 0, 0, 0
    for i in range(K):
        for j in range(K):
            var = conting_matrix[i][j]
            sum_nij += var * (var - 1) / 2.0
    sum_a = np.sum(conting_matrix, axis=0)
    sum_b = np.sum(conting_matrix, axis=1)
    for i in range(K):
        sum_ai += sum_a[i] * (sum_a[i] - 1) / 2.0
        sum_bj += sum_b[i] * (sum_b[i] - 1) / 2.0
    var = N * (N - 1) / 2.0
    ARI = (sum_nij - (sum_ai * sum_bj) / var) / (0.5 * (sum_ai + sum_bj) - sum_ai * sum_bj / var)
    print('ARI: ' + str(ARI)[:6])
    return ARI

if __name__ == "__main__":
    K = 3
    '''生成多维高斯分布数据并保存'''
    #X, Y = gaussian_data(30, K)
    # np.savetxt('GaussianData1', X)

    '''加载数据'''
    X = np.loadtxt('dataset/GaussianData')

    '''UCI数据集'''
    #X, Y = load_iris_dataset()
    '''K-means聚类 & GMM'''
    print('k-means process...')
    Y_ = kmeans_process(X, K)
    print('GMM process...')
    Y_ = GMM(X, K)
    '''对聚类结果进行评价和展示'''
    #ARI(Y, Y_)
    show(X, Y_, 'GMM')
