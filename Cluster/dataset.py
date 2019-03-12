# -*- coding: utf-8 -*-
'''
高斯分布生成k个高斯分布的数据（不同均值和方差）
'''
import numpy as np
import pandas as pd

'''
完全随机生成k个高斯矩阵
随机中心点、随机方差
num 每类的个数域
k 类别
lim x y轴位置
'''

def gaussian_data(num=30, k=3, lim=8):
    # 保存均和方差
    means = np.zeros((2))
    cov = np.zeros((2, 2))
    # 保存生成的各类高斯分布
    X = []
    Y = []
    for i in range(k):
        # 随机均值
        xc = np.random.random() * lim
        yc = np.random.random() * lim
        means[0] = xc
        means[1] = yc
        # 随机方差
        cov[0][0] = np.random.random()
        cov[1][1] = np.random.random()
        print(means, cov)
        # 产生高斯分布
        X0 = np.random.multivariate_normal(means, cov, num)
        X.append(X0)
        Y += [i for j in range(num)]
    # 随机点合并
    X0 = X[0]
    for X1 in X[1:]:
        X0 = np.concatenate((X0, X1), axis=0)
    return X0, Y

'''
Data_User_Modeling_Dataset
https://archive.ics.uci.edu/ml/datasets/User+Knowledge+Modeling
Attribute Information:
STG (The degree of study time for goal object materails),
SCG (The degree of repetition number of user for goal object materails)
STR (The degree of study time of user for related objects with goal object)
LPR (The exam performance of user for related objects with goal object)
PEG (The exam performance of user for goal objects)
UNS (The knowledge level of user, very_low, Low, Middle ,High)
'''

def load_UCI_dataset():
    # read training data and get features and labels
    training_data = pd.read_excel('dataset/Data_User_Modeling.xls', sheet_name='Training_Data')
    # read test data and get features and labels
    X, Y_lable = training_data.values[:, :-1], training_data.values[:, -1]
    Y = np.asarray(Y_lable)
    # 转换标签
    Y = np.where(Y == 'very_low', 0, Y)
    Y = np.where(Y == 'Low', 1, Y)
    Y = np.where(Y == 'Middle', 2, Y)
    Y = np.where(Y == 'High', 3, Y)

    return X, Y
    # , X_test, Y_test

def load_crytherapy_dataset():
    data = pd.read_excel('dataset/Cryotherapy.xlsx')
    X, Y = data.values[:, :-1], data.values[:, -1]
    return X, Y

def load_iris_dataset():
    # = np.loadtxt('bezdekIris.data.txt', delimiter=',',
    #                 dtype={'names': ('1', '2', '3', '4', '5'), 'formats': ('f2', 'f2', 'f2', 'f2', 'S20')})
    data = np.loadtxt('dataset/iris.txt', delimiter=',')
    X, Y = data[:, :-1], data[:, -1]

    return X, Y
