import torch
import torch.nn as nn


def corr2d(X,K):# 实现二维卷积
    h,w=K.shape
    Y=torch.zeros(size=(X.shape[0]-h+1,X.shape[1]-w+1))
    print(Y.shape)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]=torch.sum(X[i:i+h,j:j+w]*K)
    return Y
class Conv2D(nn.Module):
    def __init__(self,kernal_size):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(kernal_size))
        self.bias=nn.Parameter(torch.zeros(1))
    def forward(self,x):
        return corr2d(x,self.weight)+self.bias

# 做个边缘检测的例子
# X=torch.ones(size=(4,6))
# X[:,2:4]=0
# print(X)
# K=torch.tensor([[-1,1]])
# print(corr2d(X,K))
# y=corr2d(X,K)
# # 当然竖直方向上就没效果了
# print(corr2d(X.T,K))
# # 接下来试着通过给出原始值和特征训练出对应的W
# # conv2d=nn.Conv2d(1,1,kernel_size=(1,2),bias=False) # 实际上就相当于上面的函数 第一个个参数分别是输入通道和输出通道 后面会说
# # 参数还有很多 比如说 padding=1 也就是上下左右填充1 还可以 (1,2) 上下补充1 左右补充2
# # stride 步长
# conv2d=Conv2D((1,2))
# # X=X.reshape(1,1+X.shape)
# # y=y.reshape(1,1+X.shape)
# # 如果用了上面的就需要将其reshape
#
#
# print()
# print(X)
# print(y)
# for i in range(10):
#     y_hat=conv2d(X)
#     l=torch.pow((y_hat-y),2).sum()
#     conv2d.zero_grad()
#     l.backward()
#     conv2d.weight.data[:]-=3e-2*conv2d.weight.grad
#     print(f"epoch {i} with loss {l}")
# print(conv2d.weight)


