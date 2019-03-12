# -*- coding: utf-8 -*-
'''
共轭梯度法的普通求解版本和展示迭代过程版本
'''
import numpy as np
import math
import matplotlib.pyplot as plt
'''
共轭梯度法求最优解
:param
X, (N,m)
T, (N,1)
m,
N,
lamda, 
:return w, (m,1)
'''
def CG(X, T,lamda=0.0001):
    # 化为 Aw = b
    A = np.dot(X.T, X) + lamda*np.eye(X.shape[1])
    b = np.dot(X.T, T)
    # 初始化参数列向量
    w = np.zeros((X.shape[1], 1))
    # 初始化基向量系数
    r = b - np.dot(A, w)
    # 初始化基向量
    d = r
    rsold = np.dot(r.T, r)
    # 遍历
    for i in range(1000):
        # 求基向量系数
        alpha = rsold / np.dot(d.T, np.dot(A, d))
        # 更新参数列向量
        w = w + alpha *d
        r = r - alpha *np.dot(A, d)
        # 求取精度
        rsnew = np.dot(r.T,r)
        if math.sqrt(rsnew) < 1e-11:
            break
        # 求下一个共轭基向量
        beta = rsnew / rsold
        d = r + beta * d
        rsold = rsnew
    return w

'''
共轭梯度法求最优解, 给出迭代过程
X_ - 行向量
T_ - 行向量
首先进行转换
'''
def CG_processing(X, T,lamda=0.0001):
    plt.figure('共轭梯度法')
    # 化为 Aw = b
    A = np.dot(X.T, X) + lamda*np.eye(X.shape[1])
    b = np.dot(X.T, T)
    # 初始化参数列向量
    w = np.zeros((X.shape[1], 1))
    # 初始化基向量系数
    r = b - np.dot(A, w)
    # 初始化基向量
    d = r
    rsold = np.dot(r.T, r)
    for i in range(1000):
        # 求基向量系数
        alpha = rsold / np.dot(d.T, np.dot(A, d))
        # 更新参数列向量
        w = w + alpha *d
        r = r - alpha *np.dot(A, d)
        # 求取精度
        rsnew = np.dot(r.T,r)
        if math.sqrt(rsnew) < 1e-11:
            break
        # 求下一个共轭基向量
        beta = rsnew / rsold
        d = r + beta * d
        rsold = rsnew

        # 更新拟合曲线
        # 获取参数
        W = np.array(w[::-1]).reshape(X.shape[1])
        func = np.poly1d(W)
        xs = np.linspace(0, 1, 1000)
        y = func(xs)
        func0 = np.sin(2 * math.pi * xs)
        # 清除画板
        plt.clf()
        # 样本散点
        plt.scatter(X[:,1], T.reshape(T.shape[0]))
        # 拟合曲线
        plt.plot(xs, y, c='r')
        plt.plot(xs, func0)
        plt.xlabel('x')
        plt.ylabel('y')
        # 限制y轴范围
        plt.ylim((-1.5, 1.5))
        plt.pause(0.5)
    plt.show()
    return w
