# -*- coding: utf-8 -*-
'''
效果分析
'''
from helper_methods import *

if __name__ == '__main__':
    # 解决无法显示中文 - 指定默认字体
    plt.rcParams['font.sans-serif'] = ['FangSong']
    # 解决保存图像是负号'-'显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

    # 不同阶数下的拟合曲线和留出法求的RMSE
    #show_vary_m()

    # 不同阶数下的交叉验证法求得RMSE
    #show_vary_m_cross_validation()

    # 不同数据量下、RMSE和阶数的变化 - 交叉验证法
    #show_vary_m_in_dif_N()

    # 指定阶数和数据量确定lambda
    #show_vary_lambda()

    #梯度下降法和共轭梯度法
    # X0, T0 = generate_data(100)
    # # 拆分 5:1
    # X_, T_, X__, T__ = cross_validate(X0, T0, 5, 1)
    # X, T = deal_matrix(X_, T_, 5)
    # X_test, T_test = deal_matrix(X__, T__, 5)
    # train_GD, test_GD = show_fit_processing_GD(X, T, X_test, T_test, 0.001)
    # train_CG, test_CG = show_fit_processing_CG(X, T, X_test, T_test, 0.0001)
    # train1, = plt.plot([500*x for x in range(len(train_GD))], train_GD, c='r', ls='-')
    # test1, = plt.plot([500 * x for x in range(len(train_GD))], test_GD, c='r', ls='--')
    # train2, = plt.plot([500 * x for x in range(len(train_CG))], train_CG, c='b', ls='-')
    # test2, = plt.plot([500 * x for x in range(len(train_CG))], test_CG, c='b', ls='--')
    # plt.legend([train1, test1, train2, test2], ['GD_train_loss', 'GD_test_loss','CG_train_loss', 'CG_test_loss'])

    # 不同数据量下的拟合曲线 - 每个方法的纵向比较
    #show_fit_line_vary_Ns()

    # 四个方法的比较
    #show_compare_four_methods()

    # 四种方法的拟合曲线
    show_fitting_line_four_methods()

    plt.show()
