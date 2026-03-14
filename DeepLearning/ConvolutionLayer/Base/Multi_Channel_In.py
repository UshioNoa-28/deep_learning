from Convolution import corr2d
import torch.nn as nn
import torch
def corr2d_multi_in(X,K):# 多输入通道的卷积
    return sum(corr2d(x,k) for x,k in zip(X,K))
def corr2d_multi_in_out(X,K):
    return torch.stack([corr2d_multi_in(X,k) for k in K],0)
    # stack 指的是堆叠 这里是 对于四维权重向量K 每次拿出三维 和 输入X做多通道卷积 得到 y 再将其沿dim=0方向上堆叠
X=torch.tensor([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
X=torch.stack((X,X),dim=0) # X维数为 2 3 5
print(X)
K=torch.tensor([[1,1],[2,2]])
K=torch.stack((K,K),dim=0)# K维数为2 2 2
print(K)
print(corr2d_multi_in(X,K)) #输出为 2 4
# 接着实现多通道输入和输出
K=torch.stack((K,K,K),dim=0)# 三通道输出
print(corr2d_multi_in_out(X,K)) # 输出应该为 上面的 2 4 堆叠三次 也就是 3 2 4
# 当然框架本身也提供了类似于Linear线性层的卷积层
nn.Conv2d(in_channels=3,out_channels=1,kernel_size=(2,2),stride=(1,2),padding=(1,2))
# 直接当成Linear层用就行 不过需要注意输入要符合格式


