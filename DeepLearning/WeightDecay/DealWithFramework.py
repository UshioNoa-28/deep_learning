# 实际上torch的SGD本身允许你传入一个参数weight decay作为正则化项  下面使用参数作为简洁实现
import torch
from torch import nn
import torch.utils.data as data
import matplotlib.pyplot as plt

def plot_lambda_experiment(lambdas, train_losses, test_losses):
    plt.figure(figsize=(8,5))
    plt.plot(lambdas, train_losses, marker='o', label="train loss")
    plt.plot(lambdas, test_losses, marker='o', label="test loss")
    plt.xscale("log")
    plt.xlabel("weight_decay (L2 regularization lambda)")
    plt.ylabel("loss")
    plt.title("Effect of L2 Regularization (Framework Version)")
    plt.legend()
    plt.grid(True)
    plt.show()
n_train, n_test, nums_input, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((nums_input, 1)) * 0.01, 0.05
X_train = torch.randn((n_train, nums_input))
y_train = X_train @ true_w + true_b + torch.randn(X_train.size(0), 1) * 0.001
X_test = torch.randn((n_test, nums_input))
y_test = X_test @ true_w + true_b + torch.randn(X_test.size(0), 1) * 0.001

train_iter = data.DataLoader(data.TensorDataset(X_train, y_train), batch_size, shuffle=True)
# 准备数据 和之前一样
epochs = 100
lr = 0.01
lambdas = []
train_losses = []
test_losses = []
loss=nn.MSELoss()
lambda_range=range(-3,3)
for lambd in lambda_range:
    lambd=10**lambd
    lambdas.append(lambd)
    net=nn.Sequential(nn.Linear(in_features=nums_input,out_features=1))
    nn.init.normal_(net[0].weight,0,0.01)
    nn.init.normal_(net[0].bias,0,0.01)
    optimizer=torch.optim.SGD(net.parameters(),weight_decay=lambd,lr=lr)
    # 传入一个weight_decay即可开启正则化 你可能好奇 我没有指定开启正则化的参数 实际上它默认是全部 也就是这里其实不仅仅是W
    # 开启了正则化 b也开启了
    for epoch in range(epochs):
        for X,y in train_iter:
            l=loss(net(X),y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
    with torch.no_grad():
        l_train=loss(net(X_train),y_train)
        l_test=loss(net(X_test),y_test)
        train_losses.append(l_train.item())
        test_losses.append(l_test.item())
plot_lambda_experiment(lambdas,train_losses, test_losses)