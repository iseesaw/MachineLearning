# -*- coding: utf-8 -*-
'''
k-means聚类
'''
import numpy as np
import matplotlib.pyplot as plt

'''
X, <x1, x2, x3, ...> 样本
k, 类别
Step1. 随机初始化k个中心点
Step2. Classify 遍历每个点, 将其分为距离其最近的中心点的类别
Step3. Recenter 求每一类中每个点作为中心点的loss
'''


def kmeans(X, K):
    # 保存特征数和样本数
    n = X.shape[0]
    m = X.shape[1]
    # 保存分类结果
    Y = [0] * n
    # k个中心点
    # randomly initialize k centers
    kcenters = []
    # 在样本中随机选取中心点
    for i in range(K):
        kcenters.append(X[np.random.randint(0, n)])
    kcenters = np.asarray(kcenters)
    iters = 0
    while True:
        iters += 1
        if not iters % 10:
            print(iters)

        # 表示是否样本类别仍在改变
        is_changed = False
        # 每个类别的loss
        loss = [0] * K

        # k个类别中的点
        # 不能直接使用[[]]*k进行赋值, 可能会出错即集合元素(对象)的地址会相同
        kclass = []
        for i in range(K):
            kclass.append([])

        # 对每个样本进行分类
        for j, x in enumerate(X):
            # 求最近的中心点及类别
            c = min((np.linalg.norm(x - center), i)
                    for i, center in enumerate(kcenters))

            # 判断类别是否发生改变
            if Y[j] != c[1]:
                is_changed = True

            # 打标签
            Y[j] = c[1]
            # 分类
            kclass[c[1]].append(x)
            # 记录该中心点下的loss
            loss[c[1]] += c[0]

        if not is_changed:
            break
        # recenter
        # k个类别中重新选中心点
        for i in range(K):
            # 遍历每个点
            for xc in kclass[i]:
                # 记录该点作为中心点的loss
                temp_loss = 0
                for x in kclass[i]:
                    temp_loss += np.linalg.norm(xc - x)
                # 更新中心点
                if temp_loss <= loss[i]:
                    kcenters[i] = xc
                    loss[i] = temp_loss

    return Y, kcenters

'''
演示kmeans聚类过程
'''


def kmeans_process(X, K):
    # 保存特征数和样本数
    n = X.shape[0]
    m = X.shape[1]
    # 保存分类结果
    Y = [0] * n
    # k个中心点
    # randomly initialize k centers
    kcenters = []
    # 在样本中随机选取中心点
    for i in range(K):
        kcenters.append(X[np.random.randint(0, n)])
    kcenters = np.asarray(kcenters)

    plt.figure('K-Means Process')
    while True:
        # 表示是否样本类别仍在改变
        is_changed = False
        # 每个类别的loss
        loss = [0] * K

        # k个类别中的点
        # 不能直接使用[[]]*k进行赋值, 可能会出错即集合元素(对象)的地址会相同
        kclass = []
        for i in range(K):
            kclass.append([])

        # 对每个样本进行分类
        for j, x in enumerate(X):
            # 求最近的中心点及类别
            c = min((np.linalg.norm(x - center), i)
                    for i, center in enumerate(kcenters))

            # 判断类别是否发生改变
            if Y[j] != c[1]:
                is_changed = True

            # 打标签
            Y[j] = c[1]
            # 分类
            kclass[c[1]].append(x)
            # 记录该中心点下的loss
            loss[c[1]] += c[0]
        # recenter
        # k个类别中重新选中心点
        for i in range(K):
            # 遍历每个点
            for xc in kclass[i]:
                # 记录该点作为中心点的loss
                temp_loss = 0
                for x in kclass[i]:
                    temp_loss += np.linalg.norm(xc - x)
                # 更新中心点
                if temp_loss <= loss[i]:
                    kcenters[i] = xc
                    loss[i] = temp_loss

        # 刷新画板
        plt.clf()
        if not is_changed:
            plt.title('Over!')
        plt.scatter(X[:, 0], X[:, 1], c=Y)
        plt.scatter(kcenters[:, 0], kcenters[:, 1], marker='*', color='r')
        plt.pause(1)

        if not is_changed:
            break

    plt.show()
    return Y
