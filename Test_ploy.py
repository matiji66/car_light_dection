#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @author:Spring

# 使用matplotlib.pyplot.scatter绘制散点
import matplotlib.pyplot as plt
from pylab import mpl

# 设置默认字体，解决中文显示乱码问题
mpl.rcParams['font.sans-serif'] = ['SimHei']

# 自动计算点
x_values = list(range(1, 101))
y_values = [x ** 2 for x in x_values]
plt.scatter(x_values, y_values, s=40)

y_values = [x * 50 for x in x_values]
# 自定义颜色：c=(红色，绿色，蓝色)；取值范围：[0,1]；0深，1浅
plt.scatter(x_values, y_values, c=(1, 0, 0))

y_values = [x * 150 for x in x_values]
# 颜色映射：根据y的值，颜色由浅到深
plt.scatter(x_values, y_values, c=y_values, cmap=plt.cm.Greens)

# 设置图表标题
plt.title("平方数值表", fontsize=20)

# 设置横、纵坐标标题
plt.xlabel("数值", fontsize=12)
plt.ylabel("平方值", fontsize=12)

# 设置刻度标记大小
plt.tick_params(axis='both', labelsize=10)

# 设置每个坐标轴的取值范围[x最小，x最大，y最小，y最大]
plt.axis([0, 100, 0, 10000])

plt.show()
