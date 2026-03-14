import torch
from torch.utils import data
import torch.nn as nn
from LinearRegression import synthetic_data

true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000)
# 来看看利用pytorch框架可以帮我们简化哪些内容
def load_array(data_arrays,batch_size,is_train=True):
    dataset=data.TensorDataset(*data_arrays) # 通过输入的张量数据创建数据集 *指的是将传入的tuple拆分为多个参数
    return data.DataLoader(dataset,batch_size,shuffle=is_train)# 提供DataLoader从数据集中加载数据 可以指定batch_size数据集大小 和 shuffle 是否随机
batch_size=10
data_iter=load_array((features,labels),batch_size)
next(iter(data_iter))

# 提供简单的线性层描述
net=nn.Sequential(nn.Linear(2,1)) # 本质上是将多个layer按顺序串起来执行 和module相比更简单但是不够灵活
# net=nn.Sequential(nn.Linear(2,1),nn.ReLU(),nn.Linear(1,1)) 可以这样 然后按顺序执行


net[0].weight.data.normal_(0,0.01) # 这个指的是将自动根据第0层神经层的Linear生成对应的张量[2,1] 并用高斯分布随机填充
net[0].bias.data.fill_(0) # 这个则是使用0来初始化偏置 自动初始化为标量
print(f"original parameter {list(net[0].parameters())}")
# 损失函数和优化器也不再需要手动实现
loss=nn.MSELoss()
optimizer=torch.optim.SGD(net[0].parameters(),lr=0.02)
# 训练过程为
epochs=3
for epoch in range(epochs):
    for X,y in data_iter:
        l=loss(net(X),y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    l=loss(net(features),labels)
    print(f"epoch {epoch+1} loss {l.sum().item()}")
    print(f"epoch {epoch+1} w {net[0].weight.data}")
    print(f"epoch {epoch+1} w {net[0].bias.data}")


