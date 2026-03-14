
import torch

from PrepareImageDataset import load_data_fashion_mnist
from IPython import display
batch_size=256
train_iter,test_iter=load_data_fashion_mnist(batch_size)
# 在这里将三维转为输入向量的选择是直接拉长 但是很显然这会损失空间信息 改进方法可以看卷积神经网络
# 展平图像 图像为 单通道 28*28 也就是784个向量 且数据集有10个类别 所以 输出向量维度为10
num_input=784
num_output=10

W=torch.normal(0,0.01,size=(num_input,num_output),requires_grad=True)
b=torch.zeros(size=(num_output,),requires_grad=True)
lr=0.01
def SoftMax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(dim=1,keepdim=True)
    return X_exp/partition
def net(X):
    return SoftMax(torch.matmul(X.reshape(-1,W.shape[0]),W)+b)

# y=torch.tensor([0,2]) # 这里的0 2 代表第一类 第三类 而不是矩阵的输出形式
# y_hat=torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
# print(y_hat[[0,1],y]) # 高级索引[0,1] [0,2] 当输入为两个数组的时候 会将第一个数组的值和第二个数组的值分别作为行列坐标
# 也就是说 [0,1] [0,2] 取的坐标为 [0,0] 和 [1,2] 也就是 0.1和0.5
# 输出为0.1和0，5
# 利用这种方法就可以定义交叉熵损失函数了
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y)),y]).mean()
def accuracy(y_hat,y):
    y_hat=y_hat.argmax(axis=1) # 寻找最大值的位置 axis=0 表示沿着列查找最大值出现的位置 axis=1表示沿着行查找最大值出现的位置
    cmp=y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())
def updator(paras,lr):
    with torch.no_grad():
        for para in paras:
            para-=lr*para.grad
            para.grad.zero_()
# print(accuracy(y_hat,y)/len(y)) #正确率为 0.5 也就是第二个预测对了
def train_softmax(net,train_iter,loss,updator):
    correct = 0
    total = 0
    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)
        l.backward()
        updator((W,b),lr)
        correct+=accuracy(y_hat,y)
        total+=len(y)
    print(f"accaracy {correct/total}")
epochs=3
for epoch in range(epochs):
    train_softmax(net,train_iter,cross_entropy,updator)