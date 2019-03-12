# -*- coding: utf-8 -*-
'''
数据生成和处理函数以及效果分析辅助函数
'''
import time
from conjugate_gradient import *
from gradient_descent import *
from solve_by_analysising import *

'''
生成带高斯噪声的数据
注意生成的均是一维行向量

:return
X, (1,N) 行向量、样本值
T, (1,N) 行向量、目标值
'''
def generate_data(N=100, start=0, end=1):
    # 生成 0-1之间均匀分布的N个数
    X = np.linspace(start, end, N)
    # 对数据增加高斯噪声
    T = np.sin(2 * math.pi * X) + np.random.normal(0, 0.15, N)
    return X,T

'''
处理生成的一维行向量数据
:param
X, (1,N)
T, (1,N)

:return
X_, (N,m)
T, (N,1)

'''
def deal_matrix(X, T, m = 6):
    # 处理为需要使用的维度
    X_ = []
    for x in X:
        for i in range(m):
            X_.append(math.pow(x, i))
    X_ = np.reshape(X_, (len(X), m))
    T = T.reshape(T.shape[0], 1)
    return X_, T

'''
计算方均根差
使用矩阵运算，平方用转置矩阵相乘
X - (N,m)
T - (N,1) 目标值
w - (m,1) 参数列向量
P - (N,1) 预测值
'''
def give_performance(X, T, w):
    # 计算预测值
    P = np.dot(X, w)
    # 计算RMSE
    RMSE = np.dot((P-T).T,P-T) / X.shape[0]
    return np.round(math.sqrt(RMSE[0][0]),6)

'''
计算loss的值
X, (N,m)
T, (N,1)
w, (m,1)
1/2 * (Xw -T)'(Xw - T) + lamda/2*w'*w
'''
def give_loss(X, T, w, lamda):
    a = np.dot(X,w) - T
    loss = 0.5 * np.dot(a.T, a) + 0.5*lamda*np.dot(w.T, w)
    return np.round(math.sqrt(loss[0][0]/X.shape[0]),6)

'''
k折交叉验证
拆分数据集为测试集和训练集
X, (1,N)
T, (1,N)
t - 第t份
'''
def cross_validate(X, T, k, t):
    # 总共分成的份数
    n = int(X.size / k)
    X_train, T_train, X_test, T_test = [], [] ,[],[]
    # 份数
    for i in range(n):
        # 每份的个数
        for j in range(k):
            # 每份最后一个
            if j == k-t:
                X_test.append(X[i*k + j])
                T_test.append(T[i*k + j])
            else:
                X_train.append(X[i*k + j])
                T_train.append(T[i*k + j])
    return np.array(X_train), np.array(T_train), np.array(X_test), np.array(T_test)

'''
给出参数画出拟合曲线和原曲线
X, (N,m)
T, (N,1)
w, (m,1)
'''
def show_single_result(X, T, w, name = '拟合曲线'):
    plt.figure(name)
    # 将参数列向量转为可使用np.poly1d()的要求格式
    W = np.array(w[::-1]).reshape(X.shape[1])
    func = np.poly1d(W)
    # 取1000个点 并给出拟合曲线
    x = np.linspace(0, 1, 1000)
    y = func(x)

    # 计算方均误差，需要注意第三个参数为预测值
    RMSE = give_performance(X, T, w)
    # 给出
    plt.title('N='+str(X.shape[0])+' m=' + str(X.shape[1]) + ' RMSE=' + str(RMSE))
    # 原正弦函数
    func0 = np.sin(2 * math.pi * x)

    # 样本散点
    plt.scatter(X[:, 1], T.reshape(T.shape[0]))
    # 画出拟合曲线和原曲线
    plt.plot(x, y, c='r')
    plt.plot(x, func0)
    # 横纵轴标签
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

'''
展示拟合曲线图
使用留出法求解RMSE
:param
X, (1,N)
T, (1,N)
'''
def show_vary_m(X, T):
    m = 5
    N = 10
    X0, T0 = generate_data(N)
    # 拆分训练集和测试集
    X_train, T_train ,X_test, T_test = cross_validate(X0, T0, 5, 1)

    # 生成数据 - 训练集
    #X_train, T_train = generate_data(N)
    # 展示图
    # 阶数变化
    n = 9
    # 数据量变化
    Ns = [20, 50, 100, 500]

    # RMSE
    train_RMSEs = []
    test_RMSEs = []
    for i in range(n):
        m = i+1
        plt.subplot(431 + i)

        # 处理样本数据
        X, T = deal_matrix(X_train, T_train, m)
        X_, T_ = deal_matrix(X_test, T_test, m)

        # 无正则项
        #w = analysis_without_regular(X, T)
        # 有正则项
        w = analysis_with_regular(X, T)
        # 梯度下降法
        #w = GD(X,T)
        # 共轭梯度法
        #w = CG(X, T)

        # 获得拟合曲线函数
        W = np.array(w[::-1]).reshape(m)
        func = np.poly1d(W)
        x = np.linspace(0, 1, 1000)
        y = func(x)

        # 计算训练集MSE
        RMSE = give_performance(X, T, w)
        train_RMSEs.append(RMSE)
        # 计算测试集MSE
        RMSE1 = give_performance(X_, T_, w)
        test_RMSEs.append(RMSE1)

        # 标题
        plt.title('m=' + str(m) + ' train_RMSE=' + str(RMSE))
        func0 = np.sin(2 * math.pi * x)
        # 散点
        plt.scatter(X_train, T_train, norm=0.5, s=10)

        # 画出曲线
        l1, = plt.plot(x, y, c='r')
        l2, = plt.plot(x, func0)
        #标签
        plt.legend([l1,l2],['拟合曲线', '正弦曲线'], loc = 1)
        #横纵轴
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-1.8, 1.8))

    #plt.subplot(414)
    #plt.title('RMSE与阶数m的变化关系图')
    #train, = plt.plot([i+1 for i in range(n)], train_RMSEs)
    #test, = plt.plot([i+1 for i in range(n)], test_RMSEs, c='r')
    plt.xlabel('阶数m')
    plt.ylabel('RMSE')
    #plt.legend(handles = [train, test,], labels = ['train_RMSE', 'test_RMSE'],loc='upper center',fontsize = 'x-large' )

'''
RMSE随阶数m的变化曲线
使用k-折交叉验证求解RMSE
'''
def show_vary_m_cross_validation():
    # 数据量、最高阶数、k-折交叉验证
    Ns = [20, 50, 100, 300]
    n, k = 15, 10
    for i,N in enumerate(Ns):

        plt.subplot(221+i)
        # 生成数据
        X0, T0 = generate_data(N)
        # RMSE
        trains = []
        tests = []
        for choice in range(2):
            train_RMSEs = []
            test_RMSEs = []
            for i in range(n):
                m = i + 1
                k_train_RMSEs = []
                k_test_RMSEs = []
                for t in range(k):
                    X_train, T_train, X_test, T_test = cross_validate(X0, T0, k, t+1)
                    # 处理样本数据
                    X, T = deal_matrix(X_train, T_train, m)
                    X_, T_ = deal_matrix(X_test, T_test, m)

                    if choice == 0:
                    # 无正则项
                        w = analysis_without_regular(X, T)
                    else:
                    # 有正则项
                        w = analysis_with_regular(X, T)
                    # 梯度下降法
                    #w = GD(X,T)
                    # 共轭梯度法
                    #w = CG(X, T)

                    # 计算训练集RMSE
                    MSE = give_performance(X, T, w)
                    k_train_RMSEs.append(MSE)
                    # 计算测试集MSE
                    MSE1 = give_performance(X_, T_, w)
                    k_test_RMSEs.append(MSE1)
                train_RMSEs.append(sum(k_train_RMSEs) / k)
                test_RMSEs.append(sum(k_test_RMSEs) / k)
            trains.append(train_RMSEs)
            tests.append(test_RMSEs)

        plt.title('RMSE与阶数m的变化关系图')
        train_without, = plt.plot([i + 1 for i in range(n)], trains[0], c = 'r', ls='-')
        test_without, = plt.plot([i + 1 for i in range(n)], tests[0], c='r', ls='--')

        train_with, = plt.plot([i + 1 for i in range(n)], trains[1], c = 'b', ls='-')
        test_with, = plt.plot([i + 1 for i in range(n)], tests[1], c='b', ls = '--')

        #plt.ylim(ymin = 0)
        plt.xlabel('阶数m')
        plt.ylabel('RMSE')
        #plt.legend(handles=[train, test, ], labels=['train_RMSE', 'test_RMSE'], loc='upper center', fontsize='x-large')
        plt.legend([train_without, test_without, train_with, test_with], ['无正则项_train', '无正则项_test', '有正则项_train', '有正则项_test'])

'''
不同数据量下，RMSE和阶数的变化曲线
使用k-折交叉验证求解RMSE
'''
def show_vary_m_in_dif_N():
    # 数据量
    Ns = [20, 50, 100, 300]
    # 最高阶数/k折交叉验证
    n, k = 15, 5
    # 不同数据量下的 不同阶数的RMSE
    all_train_RMSEs = []
    all_test_RMSEs = []
    # 曲线颜色
    colors = ['b', 'r', 'g', 'y']
    # 遍历每个数据量
    for N in Ns:
        # 生成数据
        X0, T0 = generate_data(N)
        # 数据量为N时每个阶数的RMSE
        N_train_RMSEs = []
        N_test_RMSEs = []
        # 遍历lambda
        for i in range(n):
            m = i + 1
            # 每次交叉验证的RMSE、最终求平均值
            k_train_RMSEs = []
            k_test_RMSEs = []
            for t in range(k):
                # 拆分数据数据集
                X_train, T_train, X_test, T_test = cross_validate(X0, T0, k, t + 1)
                # 处理样本数据
                X, T = deal_matrix(X_train, T_train, m)
                X_, T_ = deal_matrix(X_test, T_test, m)

                # 无正则项
                #w = analysis_without_regular(X, T)
                # 有正则项
                #w = analysis_with_regular(X, T)
                # 梯度下降法
                # w = GD(X,T)
                # 共轭梯度法
                w = CG(X, T)

                # 计算训练集RMSE
                MSE = give_performance(X, T, w)
                k_train_RMSEs.append(MSE)
                # 计算测试集MSE
                MSE1 = give_performance(X_, T_, w)
                k_test_RMSEs.append(MSE1)
            # 保存数据量为N下的每个阶数的RMSE
            N_train_RMSEs.append(sum(k_train_RMSEs) / k)
            N_test_RMSEs.append(sum(k_test_RMSEs) / k)
        # 保存数据量N下的曲线
        all_train_RMSEs.append(N_train_RMSEs)
        all_test_RMSEs.append(N_test_RMSEs)

    plt.title('不同数据量时的RMSE与阶数m变化关系图')
    # 训练和测试RMSE曲线
    lines = []
    des = []
    # 标签
    for i,N in enumerate(Ns):
        # 同一数据量下的训练和测试集RMSE曲线用相同颜色、前者实线、后者虚线
        train, = plt.plot([i + 1 for i in range(n)], all_train_RMSEs[i],color=colors[i], ls='-')
        test, = plt.plot([i + 1 for i in range(n)], all_test_RMSEs[i], color=colors[i], ls='--')
        # 训练集
        lines.append(train)
        des.append(str(N)+'_train_RMSE')
        # 测试集
        lines.append(test)
        des.append(str(N)+'_test_RMSE')
    plt.xlabel('阶数m')
    plt.ylabel('RMSE')
    plt.legend(handles=lines, labels=des)

'''
指定阶数和样本数
lambda和RMSE的变化
'''
def show_vary_lambda():
    # 数据量、阶数、k-折交叉验证
    N, m, k = 20, 6, 10
    # lambda的取值
    lamda = [pow(10, -i) for i in range(1,15)]
    # 生成数据
    X0, T0 = generate_data(N)
    # 图片标题
    plt.figure()
    plt.title('m=' + str(m) + ' N=' + str(N) + ' 时lambda与RMSE的变化')
    # RMSE
    train_RMSEs = []
    test_RMSEs = []
    for l in lamda:
        # 交叉验证每次训练集和测试集RMSE
        k_train_RMSEs = []
        k_test_RMSEs = []
        for t in range(k):
            X_train, T_train, X_test, T_test = cross_validate(X0, T0, k, t + 1)
            # 处理样本数据
            X, T = deal_matrix(X_train, T_train, m)
            X_, T_ = deal_matrix(X_test, T_test, m)

            # 有正则项
            w = analysis_with_regular(X, T, l)
            # 梯度下降法
            #w = GD(X,T, l)
            # 共轭梯度法
            w = CG(X, T,l)

            # 计算训练集RMSE
            MSE = give_performance(X, T, w)
            k_train_RMSEs.append(MSE)
            # 计算测试集MSE
            MSE1 = give_performance(X_, T_, w)
            k_test_RMSEs.append(MSE1)
        train_RMSEs.append(sum(k_train_RMSEs) / k)
        test_RMSEs.append(sum(k_test_RMSEs) / k)

    # 画出关系图 其中对lambda进行取对数操作
    train, = plt.plot([math.log(x) for x in lamda], train_RMSEs)
    test, = plt.plot([math.log(x) for x in lamda], test_RMSEs, c='r')
    # 将横轴由大到小显示
    plt.gca().invert_xaxis()
    # 横纵轴
    plt.xlabel('ln(lambda)')
    plt.ylabel('RMSE')
    #plt.ylim(ymin = 0)
    plt.legend([train,test], ['train_RMSE', 'test_RMSE'], loc = 1)

'''
梯度下降自适应步长
步长衰减次数和梯度的变化曲线
'''
def show_gradient_vary_step(gra):
    plt.figure('初始步长为1, 衰减率为0.5时梯度随步长衰减次数的变化')
    plt.title('初始步长为1, 衰减率为0.5时梯度随步长衰减次数的变化')
    plt.plot(range(0, len(gra)), gra, c='r')
    for i, g in enumerate(gra):
        plt.text(i, g, '%.6f' % math.pow(0.5, i), ha='center', va='bottom', fontsize=9)
    plt.xlabel('步长衰减次数')
    plt.ylabel('梯度')
    plt.show()

'''
梯度下降迭代次数和训练集、测试集loss变化
'''
def show_fit_processing_GD(X, T,X_test, T_test, lamda):
    # 初始化参数列向量
    w = np.zeros((X.shape[1], 1))
    # 梯度最大距离/步长衰减率/初始步长
    dis, step = 10.0, 0.001
    # processing_data = []
    # 判断是否所有梯度距离都比精度小
    iterations = 0
    trains_loss, tests_loss = [], []
    while dis > 1e-4:
        # 求解梯度
        w0 = np.dot(np.dot(X.T, X) + lamda * np.eye(w.size), w) - np.dot(X.T, T)
        # 下降距离
        w1 = step * w0
        w = w - w1
        # 求梯度距离列向量中的最大值
        dis = np.linalg.norm(w1, np.inf)
        iterations = iterations + 1
        if iterations % 500 == 0:
            train_loss = give_loss(X, T, w, lamda)
            test_loss = give_loss(X_test, T_test, w, lamda)
            trains_loss.append(train_loss)
            tests_loss.append(test_loss)
    return trains_loss, tests_loss

'''
共轭梯度迭代次数和训练集、测试集loss变化
'''
def show_fit_processing_CG(X, T,X_test, T_test, lamda):
    # 化为 Aw = b
    A = np.dot(X.T, X) + lamda * np.eye(X.shape[1])
    b = np.dot(X.T, T)
    # 初始化参数列向量
    w = np.zeros((X.shape[1], 1))
    # 初始化基向量系数
    r = b - np.dot(A, w)
    # 初始化基向量
    d = r
    rsold = np.dot(r.T, r)

    trains_loss, tests_loss = [], []
    # 遍历
    for i in range(1000):
        # 求基向量系数
        alpha = rsold / np.dot(d.T, np.dot(A, d))
        # 更新参数列向量
        w = w + alpha * d
        r = r - alpha * np.dot(A, d)
        # 求取精度
        rsnew = np.dot(r.T, r)
        if math.sqrt(rsnew) < 1e-11:
            break
        # 求下一个共轭基向量
        beta = rsnew / rsold
        d = r + beta * d
        rsold = rsnew

        train_loss = give_loss(X, T, w, lamda)
        test_loss = give_loss(X_test, T_test, w, lamda)
        trains_loss.append(train_loss)
        tests_loss.append(test_loss)
    return trains_loss, tests_loss

'''
不同数据量下的拟合曲线
使用221布局
'''
def show_fit_line_vary_Ns():
    Ns = [10, 50, 100, 300]
    m = 6

    plt.suptitle('拟合曲线')
    for i,N in enumerate(Ns):
        plt.subplot(221+i)
        X0, T0 = generate_data(N)
        X, T = deal_matrix(X0, T0, m)

        #w = analysis_without_regular(X, T)
        #w = analysis_with_regular(X, T)
        #w = GD(X, T)
        w = CG(X,T)
        # 获得拟合曲线函数
        W = np.array(w[::-1]).reshape(m)
        func = np.poly1d(W)
        x = np.linspace(0, 1, 1000)
        y = func(x)

        loss = give_loss(X, T, w, 0.001)
        # 标题
        plt.title('m=' + str(m) + ' loss=' + str(loss))
        func0 = np.sin(2 * math.pi * x)
        # 散点
        plt.scatter(X0, T0, norm=0.5, s=10)
        # 画出曲线
        l1, = plt.plot(x, y, c='r')
        l2, = plt.plot(x, func0)
        # 标签
        plt.legend([l1, l2], ['拟合曲线', '正弦曲线'], loc=1)
        # 横纵轴
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-1.8, 1.8))

'''
四种方法对比的代码框架
同数据量、同lambda、RMSE随阶数变化
同数据量、同阶数、RMSE随lambda变化
同阶数、同lambda、RMSE随不同数据量的变化
(为了完成相应的对比, 会在代码内部进行相应的修改)
'''
def show_compare_four_methods():
    m = 6
    lamda = 0.001
    Ns = [10, 50, 100, 300, 500]
    trains_0, tests_0 = [], []
    trains_1, tests_1 = [], []
    trains_2, tests_2 = [], []
    trains_3, tests_3 = [], []
    for N in Ns:
        X0, T0 = generate_data(N)
        X_tr, T_tr, X_te, T_te = cross_validate(X0, T0, 5, 1)
        X_train, T_train = deal_matrix(X_tr, T_tr, m)
        X_test, T_test = deal_matrix(X_te, T_te, m)
        for methods in range(4):
            if methods == 0:
                w = analysis_without_regular(X_train, T_train)
                train_RMSE = give_performance(X_train, T_train, w)
                test_RMSE = give_performance(X_test, T_test, w)
                trains_0.append(train_RMSE)
                tests_0.append(test_RMSE)
            if methods == 1:
                w = analysis_with_regular(X_train, T_train, lamda)
                train_RMSE = give_performance(X_train, T_train, w)
                test_RMSE = give_performance(X_test, T_test, w)
                trains_1.append(train_RMSE)
                tests_1.append(test_RMSE)
            if methods == 2:
                w = GD(X_train, T_train, lamda)
                train_RMSE = give_performance(X_train, T_train, w)
                test_RMSE = give_performance(X_test, T_test, w)
                trains_2.append(train_RMSE)
                tests_2.append(test_RMSE)
            if methods == 3:
                w = CG(X_train, T_train, lamda)
                train_RMSE = give_performance(X_train, T_train, w)
                test_RMSE = give_performance(X_test, T_test, w)
                trains_3.append(train_RMSE)
                tests_3.append(test_RMSE)

    train0, = plt.plot(Ns, trains_0, c='y', ls='-')
    test0, = plt.plot(Ns, tests_0, c='y', ls='--')

    train1, = plt.plot(Ns, trains_1, c='b', ls='-.')
    test1, = plt.plot(Ns, tests_1, c='b', ls='--')

    train2, = plt.plot(Ns, trains_2, c='g', ls='-')
    test2, = plt.plot(Ns, tests_2, c='g', ls='--')

    train3, = plt.plot(Ns, trains_3, c='r', ls='--')
    test3, = plt.plot(Ns, tests_3, c='r', ls=':')
    plt.xlabel('N')
    plt.ylabel('RMSE')
    #plt.legend([train0, test0, train1, test1, train2, test2, train3, test3], ['解析解无正则_train', '解析解无正则_test','解析解有正则_train', '解析解有正则_test', '梯度下降法_train', '梯度下降法_test', '共轭梯度法_train', '共轭梯度法_test'])

'''
指定阶数、lambda等参数
展示四种方法在不同数据量下的拟合曲线
'''
def show_fitting_line_four_methods():
    m = 6
    lamda = 0.001
    N = 100
    X0, T0 = generate_data(N)
    X_tr, T_tr, X_te, T_te = cross_validate(X0, T0, 5, 1)
    X, T = deal_matrix(X_tr, T_tr, m)
    X_test, T_test = deal_matrix(X_te, T_te, m)
    plt.suptitle('四种方法在m='+str(m)+' N='+str(N)+' lambda='+str(lamda)+'时拟合曲线')
    methods = ['解析解无正则项', '解析解有正则项','梯度下降法', '共轭梯度法']
    for i, method in enumerate(methods):
        plt.subplot(221+i)
        if i ==0:
            w = analysis_without_regular(X,T)
        if i == 1:
            w = analysis_with_regular(X,T)
        if i == 2:
            w = GD(X,T)
        if i == 3:
            w = CG(X,T)
        W = np.array(w[::-1]).reshape(X.shape[1])
        func = np.poly1d(W)
        # 取1000个点 并给出拟合曲线
        x = np.linspace(0, 1, 1000)
        y = func(x)
        # 计算方均误差，需要注意第三个参数为预测值
        RMSE = give_performance(X_test, T_test, w)
        # 给出
        plt.title(method + ' 测试集RMSE=' + str(RMSE))
        # 原正弦函数
        func0 = np.sin(2 * math.pi * x)
        # 样本散点
        plt.scatter(X[:, 1], T.reshape(T.shape[0]), s=10)
        # 画出拟合曲线和原曲线
        plt.plot(x, y, c='r')
        plt.plot(x, func0)
        # 横纵轴标签
        plt.xlabel('x')
        plt.ylabel('y')
