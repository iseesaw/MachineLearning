# -*- coding: utf-8 -*-
from dataset import *
from logistic_regression import *
import matplotlib.pyplot as plt

'''
生成高斯分布数据集
四种方法的逻辑回归可视化
'''

def logistics_regression():
    methods = ['Gradient Descent',
               'Gradient Descent + regularlization',
               'Newtons',
               'Newtons + regularlization']
    # 生成数据
    N, prop = 200, 0.4
    X, Y = binary_label_data(N, prop)
    X = add_one_column(X)
    # 测试集
    X_test, Y_test = binary_label_data(int(N * 0.3), prop)
    X_test = add_one_column(X_test)
    # 调用logistics regression
    # W1 = LR_GD(X, Y, 0)
    # W2 = LR_GD(X, Y, 0.001)
    # W3 = LR_Newtons(X, Y)
    # W4 = LR_Newtons(X, Y, 0.0001)
    # Ws = [W1, W2, W3, W4]
    plt.figure()
    # plt.title("Logistics Regression")
    # for i in range(4):
    #     plt.subplot(221 + i)
    #     W = Ws[i]
    #     # 评分
    #     p, r, F1 = logistics_score(X_test, Y_test, W)
    #     # 获得决策曲线
    #     x0 = np.linspace(min(X[:, 1]), max(X[:, 1]), 1000)
    #     y0 = (W[1][0] * x0 + W[0][0]) / (-W[2][0])
    #     # 画点和分类线
    #     plt.title(methods[i] + "\n" + "Precision=" + str(p)[:6] + " Recall=" + str(r)[:6] + " F1=" + str(F1)[:6])
    #     plt.scatter(X[:, 1], X[:, 2], c=Y.reshape(N), s=15)
    #     plt.plot(x0, y0)

    W = LR_Newtons(X, Y)
    p, r, F1 = logistics_score(X_test, Y_test, W)
    # 获得决策曲线
    x0 = np.linspace(min(X[:, 1]), max(X[:, 1]), 1000)
    y0 = (W[1][0] * x0 + W[0][0]) / (-W[2][0])
    # 画点和分类线
    plt.title(methods[2] + "\n" + "Precision=" + str(p)[:6] + " Recall=" + str(r)[:6] + " F1=" + str(F1)[:6])
    plt.scatter(X[:, 1], X[:, 2], c=Y.reshape(N), s=15)
    plt.plot(x0, y0)

    plt.show()

'''
UCI冻伤数据集
逻辑回归分类
'''

def logistics_uci_dateset():
    # 加载冻伤数据集
    X, Y = load_cryotherapy_data()
    # 加工X特征集
    X = add_one_column(X)
    # 切分训练集和测试集
    X_train, X_test = X[:-30, ], X[-30:, ]
    Y_train, Y_test = Y[:-30, ], Y[-30:, ]
    W = LR_Newtons(X_train, Y_train, 0.0001)
    # W = LR_regular_GD(X_train, Y_train)
    # W = LR_no_regular_GD(X_train, Y_train)
    # 评价得分
    p,r,F1 = logistics_score(X_test, Y_test, W)
    print("Precision=" + str(p)[:6] + " Recall=" + str(r)[:6] + " F1=" + str(F1)[:6])


'''
X为加一列之后的数据集
逻辑回归性能
样本\预测   正例   反例
正例         TP     FN
反例         FP     TN
accuracy = TP+TN / N + P
precision = TP / TP + FP
recall = TP / TP + FN
F1 = 2*precision*recall / (precision + recall)
'''

def logistics_score(X, Y, W):
    # 计算正例样本总数
    positive_n = np.count_nonzero(Y)
    # 　预测分类结果
    prob_predict = sigmoid(np.dot(X, W))
    TP, FP = 0, 0
    for prob, y in zip(prob_predict, Y):
        if prob >= 0.5 and y:
            TP += 1
        elif prob >= 0.5 and not y:
            FP += 1

    precision = TP / (TP + FP)
    recall = TP / positive_n
    F1 = 2 * precision * recall / (precision + recall)
    return precision, recall, F1
    # print("precision: ", precision)
    # print("recall: ", recall)

'''
maxsoft 三类别分类可视化
'''

def maxsoft_classification():
    N = 300
    X, Y = multi_label_data(N)
    # 数据集加工, 适合模型
    X0 = add_one_column(X)
    Y0 = hot_code(Y, 3)
    W = multi_LR(X0, Y0)

    # 测试集
    X_test, Y_test = multi_label_data(int(N * 0.3))
    X_test = add_one_column(X_test)
    # 评分
    maxsoft_score(X_test, Y_test, W)
    # 可视化
    x0 = np.linspace(0, 6, 1000)
    y0 = (W[1][0] * x0 + W[0][0]) / (-W[2][0])
    y1 = (W[1][1] * x0 + W[0][1]) / (-W[2][1])
    y2 = (W[1][2] * x0 + W[0][2]) / (-W[2][2])
    # 画点和分类线
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=Y.reshape(N), s=15)
    plt.plot(x0, y0)
    # plt.plot(x0, y1)
    plt.plot(x0, y2)
    plt.xlim((min(X[:, 0]) - 1, max(X[:, 1]) + 1))
    plt.show()

'''
maxsoft分类mnist数据集
'''

def maxsoft_mnist_set():
    X_train, Y_train, X_test, Y_test = load_mnist_set()
    W = multi_LR(X_train, Y_train)
    maxsoft_score(X_test, Y_test, W)
    # 保存mnist模型参数
    np.savetxt("mnist_model", W)

'''
加1列1
'''

def add_one_column(X0):
    n = X0.shape[0]
    # 补齐
    X = np.ones((n, 1))
    X = np.concatenate((X, X0), axis=1)
    return X

'''
X 第0列插入1用来补齐W
Y 标签用热码编码 0-100, 1-010, 2-001
'''

def hot_code(Y0, k):
    n = Y0.shape[0]
    Y = np.zeros((n, k))
    # 重编码Y
    # 1 - 1, 0, 0; 2- 0, 1, 0; 3- 0, 0, 1
    for i in range(Y.shape[0]):
        Y[i][int(Y0[i][0])] = 1
    return Y

'''
加载mnist手写字体数据集并加工
'''

def load_mnist_set():
    X_train, Y_train = load_image_set("mnist\\train-images.idx3-ubyte"), load_label_set(
        "mnist\\train-labels.idx1-ubyte")
    X_test, Y_test = load_image_set("mnist\\t10k-images.idx3-ubyte"), load_label_set("mnist\\t10k-labels.idx1-ubyte")
    # reshape
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2])
    # 归一化, 避免乘法溢出 exp(x1)
    X_train = X_train
    X_test = X_test
    # 数据加工, 升维, 标签转为热码
    # 训练集加工
    X_train = add_one_column(X_train)
    Y_train = hot_code(Y_train, 10)
    # 　测试集加工, 注意测试集标签不用加工, 方便直接验证
    X_test = add_one_column(X_test)
    return X_train / 255, Y_train, X_test / 255, Y_test

'''
X (n, m), Y (n, 1), W (m, k)
多分类得分
'''

def maxsoft_score(X, Y, W):
    # 计算测试集样本各类别概率得出分类结果
    Y_precdict = np.argmax(np.dot(X, W), axis=1)
    accuracy_num = 0
    # 计算测试集分类正确个数
    for i in range(Y.shape[0]):
        if Y[i][0] == Y_precdict[i]:
            accuracy_num += 1
    print("Acccuarcy: " + str(accuracy_num / Y.shape[0]))

if __name__ == "__main__":
    #logistics_regression()
    logistics_uci_dateset()
    # maxsoft_classification()
    # maxsoft_mnist_set()
