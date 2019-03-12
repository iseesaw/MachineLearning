# -*- coding: utf-8 -*-
'''
实现解析解求解方法（有正则式、无无正则式）
'''
import numpy as np
'''
解析解求解、无正则项
X, (N,m)
T, (N,1)
m, 阶数
N, 样本数
:return w, (m,1) 参数列向量
'''
def analysis_without_regular(X, T):
    # 求逆
    r = np.linalg.inv(np.dot(X.T, X))
    w0 = np.dot(r, X.T)
    w = np.dot(w0, T)
    return w

'''
解析解求解、带正则项
X, (N,m)
T, (N,1)
m, 阶数
N, 样本数
lam, 正则项系数
:return w, (m,1) 参数列向量
'''
def analysis_with_regular(X, T,lam = 0.001):
    r = np.dot(X.T, X) + lam * np.eye(X.shape[1])
    w0 = np.dot(np.linalg.pinv(r), X.T)
    w = np.dot(w0, T)
    return w
