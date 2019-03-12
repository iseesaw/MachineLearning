# -*- coding: utf-8 -*-
'''
混合高斯模型
E
M
'''
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from kmeans import kmeans
from init import ARI
'''
X, n samples
K, number of classes
'''

def GMM(X, K):
    # n samples, m features
    N, M = X.shape[0], X.shape[1]

    # N*K, prob of Xi belongs to Ck
    # N*K
    W = np.ones((N, K)) * 1. / K;

    # prob of each class
    # K*1
    PI = np.ones((K, 1)) * 1. / K

    # means
    # K*M
    # kmeans initization
    var, U= kmeans(X, K)
    # U = np.asarray([X[np.random.randint(0, N)] for i in range(K)]).reshape((K, M))
    # cov matrix
    # K*M*M
    E = np.tile(np.eye(M), (K, 1)).reshape((K, M, M))

    iters = 0;
    MaxIters = 20;

    # plt.figure()
    while iters < MaxIters:
        # avoid singular matrix
        E = np.multiply(E, np.tile(np.eye(M), (K, 1)).reshape((K, M, M)))
        # if not iters % 10:
        getL(X, U, E, PI)
        ##### Expect ######
        for i in range(N):
            for j in range(K):
                W[i][j] = st.multivariate_normal(U[j], E[j]).pdf(X[i]) * PI[j]
            # normalization
            W[i] /= np.sum(W[i])

        ##### Maxilize #####
        PI = (1. / N) * np.sum(W, axis=0).reshape((K, 1))

        # update means
        for j in range(K):
            U[j] = 0
            for i in range(N):
                U[j] += W[i][j] * X[i]
            U[j] /= np.sum(W, axis=0)[j]

        # update cov matrix
        for j in range(K):
            E[j] = 0
            for i in range(N):
                E[j] += W[i][j] * np.dot(X[i] - U[j], (X[i] - U[j]).T)
            item = np.sum(W, axis=0)[j]
            E[j] /= item

        iters += 1
        if not iters % 10:
            print(iters)
            # Y = np.argmax(W, axis=1)
            # # 刷新画板
            # plt.clf()
            # plt.scatter(X[:, 0], X[:, 1], c=Y)
            # plt.pause(1)

    # plt.show()

    # get the result of classification
    Y = np.argmax(W, axis=1)
    return Y

def getL(X, U, E, PI):
    N, M, K = X.shape[0], X.shape[1], U.shape[0]
    L = 0
    for i in range(N):
        S = 0
        for j in range(K):
            S += st.multivariate_normal(U[j], E[j]).pdf(X[i]) * PI[j]
        L += np.log(S)
    print(L)
