# -*- coding: utf-8 -*-
'''
实现梯度下降法的普通版、以及迭代过程可视化版
'''
import math
import numpy as np
import matplotlib.pyplot as plt
'''
梯度下降法求最优解
使用自适应步长、衰减系数为0.5
:param
X, (N,m)
T, (N,1)
lamda, 正则项系数
step, 步长
:return w, (m,1)
'''
def GD(X, T, lamda=0.0001):
    # 初始化参数列向量
    w = np.zeros((X.shape[1], 1))
    # 梯度最大距离/步长衰减率/初始步长
    dis, step = 10.0, 0.001
    #ar = 0.5
    # 判断是否所有梯度距离都比精度小
    iterations = 0
    loss_processing = []
    # 数量越多、精度应该搞点
    while dis > 1e-4:
        # 求解梯度
        w0 =np.dot(np.dot(X.T, X) + lamda*np.eye(w.size),w) - np.dot(X.T, T)
        # 下降距离
        w1 = step * w0

        # # 自适应步长求解
        # new_w = w - w1
        # new_w0 = np.dot(np.dot(X.T, X) + lamda * np.eye(w.size), new_w) - np.dot(X.T, T)
        # # 如果梯度增加、则步长衰减
        # while np.linalg.norm(new_w0, 2) >= np.linalg.norm(w0, 2):
        #     # 步长衰减
        #     step = step*ar
        #     # 求取步长衰减后的梯度距离
        #     w1 = step*w0
        #     # 求取步长衰减后的参数列向量
        #     new_w = w - w1
        #     # 求取新的梯度、并进行循环判断
        #     new_w0 = np.dot(np.dot(X.T, X) + lamda*np.eye(w.size),new_w) - np.dot(X.T, T)
        #     # 保存衰减过程的梯度
        #     #processing_data.append(math.sqrt(np.linalg.norm(new_w0, 2)))

        # 更新参数列向量
        w = w - w1
        # 求梯度距离列向量中的最大值
        dis = np.linalg.norm(w1, np.inf)
        iterations = iterations + 1
        if iterations % 500 == 0:
            a = np.dot(X, w) - T
            loss = 0.5 * np.dot(a.T, a) + 0.5 * lamda * np.dot(w.T, w)
            loss_  = np.round(loss[0][0] / X.shape[0], 6)
            loss_processing.append(loss_)
    return w
    #, processing_data

'''
梯度下降法求最优解, 给出迭代过程
:param
X, (N,m)
T, (N,1)
lamda, step
:return w, (m,1)
'''
def GD_processing(X, T, lamda=0.0001, step = 0.001):
    # 初始化参数列向量
    w = np.zeros((X.shape[1], 1))
    # 梯度最大距离
    dis = 10.0
    # 迭代次数
    batch = 0
    plt.figure('梯度下降法')
    # 判断是否所有梯度距离都比精度小
    while dis > 1e-4:
        # 求解梯度
        w0 =np.dot(np.dot(X.T, X) + lamda*np.eye(w.size),w) - np.dot(X.T, T)
        # 下降距离
        w1 = step * w0
        # 更新参数列向量
        w = w - w1
        # 求梯度距离列向量中的最大值
        dis = np.linalg.norm(w1, np.inf)
        batch =batch + 1
        # 每1000次迭代更新一次图片
        if batch % 1000 == 0 :
            print(batch)
            # 将(m,1)维参数w 改成(1,m)并反序
            W = np.array(w[::-1]).reshape(X.shape[1])
            # 拟合曲线函数
            func = np.poly1d(W)
            # 取1000个样本点
            x = np.linspace(0, 1, 1000)
            y = func(x)
            func0 = np.sin(2 * math.pi * x)
            plt.clf()
            # 数据散点
            plt.scatter(X[:,1], T.reshape(T.shape[0]))
            # 拟合曲线
            plt.plot(x, y, c='r')
            plt.plot(x, func0)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.pause(0.01)
    plt.show()
    return w
