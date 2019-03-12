# -*- coding: utf-8 -*-
'''
展示四种方法的拟合结果和拟合过程
'''
from helper_methods import *

if __name__ == '__main__':
	# 解决无法显示中文 - 指定默认字体
	plt.rcParams['font.sans-serif'] = ['FangSong']
	# 解决保存图像是负号'-'显示为方块的问题
	plt.rcParams['axes.unicode_minus'] = False
	# 产生数据
	X0, T0 = generate_data()
	# 处理数据
	X, T = deal_matrix(X0, T0)

	# 展示解析解拟合曲线
	# 无正则项
	w1 = analysis_without_regular(X, T)
	# 展示无正则项解析解
	show_single_result(X, T, w1,'解析解(无正则项)')

	# 有正则项
	w2 = analysis_with_regular(X, T)
	# 展示有正则项解析解
	show_single_result(X, T, w1, '解析解(有正则项)')

	# 梯度下降法拟合过程
	GD_processing(X, T)

	# 共轭梯度法拟合过程
	CG_processing(X, T)
