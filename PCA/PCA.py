# -*- coding: utf-8 -*-
import numpy as np

'''
主成分分析
将X降维为K维
'''

def PCA(X, K, normal=False):
    # N samples, M features
    N, M = X.shape[0], X.shape[1]

    # 降维后的样本 N*K
    X_ = np.zeros((N, K))

    # 计算每个特征的期望值并对样本集进行归一化
    # 按列求和并取平均
    U = np.sum(X, axis=0) / N
    # 不能广播, 故需要将求取的期望进行扩展
    X = X - U.reshape((1, M)).repeat(N, axis=0)
    # ***使用库函数***
    # U = np.mean(X, axis=0)
    # X = X - U

    # 计算每个特征的方差并对样本集进行归一化
    # 当样本集是图像时不需要进行方差归一化
    if normal:
        # 逐元素元素将每个样本特征值平方
        X2 = np.power(X, 2)
        # 按列求和并取平均, 开方得到方差
        cov = np.sqrt(np.sum(X2, axis=0) / N)
        # 先扩展再归一化
        X = X / cov.reshape((1, M)).repeat(N, axis=0)

    # 计算协方差矩阵
    E = np.zeros((M, M))
    # 求和 XX^T
    for i in range(N):
        E = E + np.dot(np.asarray(X[i]).reshape((M, 1)), np.asarray(X[i]).reshape((1, M)))
    # 取平均
    E = E / (N-1)
    # ***使用库函数***
    #E = np.cov(X, rowvar=0)

    ##############特征值分解############
    # 直接求解特征值和特征向量
    # lams, Us = np.linalg.eig(E)
    # # 使用'-'倒序
    # indexs = np.argsort(-lams)
    # # 返回的特征向量是列向量, 取Topk列即前K个特征值
    # topk_Us = Us[:, indexs[:K]]
    #
    # # 降维
    # X_ = np.dot(X, topk_Us)
    # # 重建
    # re_X = np.dot(X_, topk_Us.T) + U
    # # 由于使用eig函数计算返回的是复数形式
    # # 因此需要提取实数部分
    # re_X = np.real(re_X)

    ############SVD分解################
    #SVD分解特征值
    W, singulars, V = np.linalg.svd(E)
    # 降维
    X_ = np.dot(X, W[:, :K])
    # 重建
    re_X = np.dot(X_, W[:, :K].T) + U
    # 返回降维后样本矩阵和重建的样本矩阵
    return X_, re_X
