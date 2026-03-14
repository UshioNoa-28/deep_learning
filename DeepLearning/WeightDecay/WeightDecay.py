# 来个例子展示权重衰退对模型泛用性的作用
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt


def plot_lambda_experiment(lambdas, train_losses, test_losses):
    plt.figure(figsize=(8,5))
    plt.plot(lambdas, train_losses, marker='o', label="train loss")
    plt.plot(lambdas, test_losses, marker='o', label="test loss")
    plt.xscale("log")  # λ通常用对数尺度
    plt.xlabel("lambda (L2 regularization)")
    plt.ylabel("loss")
    plt.title("Effect of L2 Regularization")
    plt.legend()
    plt.grid(True)
    plt.show()
n_train,n_test,nums_input,batch_size=20,100,200,5
lr,lamuda_range=0.01,range(-3,3)
true_w,true_b=torch.ones((nums_input,1))*0.01,0.05
X_train=torch.randn((n_train,nums_input))
y_train=X_train@true_w+true_b
y_train+=torch.randn(y_train.size())*0.001
X_test=torch.randn(n_test,nums_input)
y_test=X_test@true_w+true_b
y_test+=torch.randn(y_test.size())*0.001
train_dataset=data.TensorDataset(X_train,y_train)
test_dataset=data.TensorDataset(X_test,y_test)
train_iter=data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_iter=data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
# 生成数据 对应的关系是为 y=0.01(x1+x2+...)+0.05
def l2_penalty(W): # L2 范数惩罚
    return torch.sum((W*W))/2
def linear(X):
    return X@W+b
def mSELoss(y_hat,y):
    return torch.mean(torch.pow((y_hat-y),2))/2

epochs=100
lambdas=[]
train_losses=[]
test_losses=[]
for lamud in lamuda_range:
    W = torch.normal(0, 0.01, size=true_w.size(), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.SGD((W, b), lr=lr)
    lamuda=10**lamud # 构造从10^-3 到 10 λ的图像
    lambdas.append(lamuda) # 收集lambda生成图像
    for epoch in range(epochs):
        for X,y in train_iter:
            y_hat=linear(X)
            l=(mSELoss(y_hat,y)+lamuda*l2_penalty(W))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
    print("true_w norm:", torch.norm(true_w).item())
    print("learned_w norm:", torch.norm(W).item())
    with (torch.no_grad()): # 测试的时候不再需要加上正则项了 本身就是用来训练的时候用的东西
        l_train=mSELoss(linear(X_train),y_train)
        l_test=mSELoss(linear(X_test),y_test)
        train_losses.append(l_train.item())
        test_losses.append(l_test.item())
plot_lambda_experiment(lambdas,train_losses,test_losses)
# 通过参数来观察学习
# 结果可以见res 随着λ逐渐上升 模型的泛化能力增强 但是训练集的损失逐渐增加
# 上升到100的时候 约束过大 导致必要的参数都不能被学习 导致泛化能力和训练集上的准确度一起下降
