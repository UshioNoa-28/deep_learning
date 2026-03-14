# 实际上为加强版的LeNet 主要体现在卷积层数量增加
# 激活函数变为Relu 使用dropout和数据增强 这个cpu已经跑不动了 所以不要在这里运行
import torch
import torch.nn as nn

net=nn.Sequential(
    nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Conv2d(96,256,kernel_size=5,padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Conv2d(256,384,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.Conv2d(384,384,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.Conv2d(384,256,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Flatten(),
    nn.Linear(6400,4096),
    nn.ReLU(),nn.Dropout(0.5),
    nn.Linear(4096,4096),
    nn.ReLU(),nn.Dropout(0.5),
    nn.Linear(4096,10)
)
X=torch.rand(size=(1,1,224,224))
for layer in net:
    X = layer(X)
    print(f"{layer.__class__.__name__} {X.shape}")
# 不再展示额外内容