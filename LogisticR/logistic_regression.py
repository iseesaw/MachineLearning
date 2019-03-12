# -*- coding: utf-8 -*-
'''
Logistic Regression
'''
import numpy as np

# 使用梯度下降法
def LR_GD(X, Y, lam=0):
    print("Gradient Descent lam=" + str(lam) + "...")
    n, m = X.shape[0], X.shape[1]
    # 初始化
    W = np.ones((m, 1))
    # 步长 正则
    alpha = 0.5
    # 最大迭代次数
    iterations = 0
    while iterations < 500:
        # 估计类别和实际类别误差
        E = Y - sigmoid(np.dot(X, W))
        # 梯度
        gradient = np.dot(X.T, E)
        # 梯度上升
        W = W + alpha / n * gradient + lam / n * W
        iterations += 1
        if not iterations % 100:
            error = 1 / n * np.sqrt(np.linalg.norm(E, 2))
            print(iterations, error)
    return W

# 牛顿法
# 　在似然函数前加负号, 则求似然函数最小化
def LR_Newtons(X, Y, lam=0):
    print("Newtons Method lam=" + str(lam) + "...")
    # 获取维度信息
    n, m = X.shape[0], X.shape[1]
    # 初始化
    W = np.zeros((m, 1))
    # 正则项系数
    # 最大迭代次数
    iterations = 0
    while iterations < 500:
        # logistic function
        logistic = sigmoid(np.dot(X, W))
        # 一阶导
        first_der = -np.dot(X.T, Y - logistic)
        # 获得对角线为 p(1-p)的对角矩阵
        A = np.dot(logistic, 1 - logistic.T)
        A = np.multiply(np.eye(A.shape[0]), A)
        # 　海森矩阵　X'AX
        hession = np.dot(np.dot(X.T, A), X)
        # 更新
        W = W - 1 / n * np.dot(np.linalg.inv(hession), first_der) + lam / n * W
        iterations += 1
        if not iterations % 100:
            error = 1 / n * np.sqrt(np.linalg.norm(Y - logistic, 2))
            print(iterations, error)
    return W

# logistic function
def sigmoid(XW):
    return 1.0 / (1 + np.exp(-XW))

# softmax 多分类逻辑回归
# dataset Mnist Iris
# n个样本, m个属性(包括补齐的1), k个类别
def multi_LR(X, Y):
    n, m, k = X.shape[0], X.shape[1], Y.shape[1]
    # 初始化权重 (m, k)
    W = np.ones((m, k))
    # 步长
    alpha = 0.5
    lam = 0.00001
    # 迭代次数
    iterations = 0
    error = 1
    while iterations < 300:
        # softmax, 求每个样本分别属于每个类别的概率
        softmax_ = softmax(np.dot(X, W))
        # 梯度
        gradient = np.dot(X.T, (softmax_ - Y))
        # 更新W, 除以n归一化, lam正则化
        W = W - alpha / n * gradient + lam / n * W
        # 预测结果与实际结果的差值
        iterations += 1
        if not iterations % 50:
            error = 1 / n * np.sqrt(np.linalg.norm(softmax_ - Y, 2))
            print(iterations, error)

    return W

'''
@:param
    样本数n, 特征数m, 类别数k
    X (n, m)
    W (m, k)
    XW (n, k), 
@:return
    (n, k), 每个样本属于每一类的概率

'''

def softmax(XW):
    # 　注意对数据进行归一化, 避免指数溢出
    prob = np.exp(XW)
    # 求每个样本属于各个类别的概率
    # 横向求和并扩展
    sum_prob = np.sum(prob, axis=1).reshape((prob.shape[0], 1)).repeat(XW.shape[1], axis=1)
    return prob / sum_prob
