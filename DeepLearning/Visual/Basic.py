# 展示一下最简单的数据可视化 通过matplotlib.pyplot 实现
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[2,4,6,8,10]
# 需要注意的是 画图的函数不支持传入tensor
plt.figure(figsize=(5,8))# 创建画布 figsize为画布的大小
# 折线图用plot 散点图用scatter
plt.plot(x,y,label="my_line",color="red",marker="o") # label用来标记这条线的名字 color为线的颜色 marker为标记点的形状 o 表示圆圈
plt.xlabel("odd")
plt.ylabel("even")
plt.legend() # 显示图例 会在左上角标上 红线 my_line
plt.grid(True)
plt.show()