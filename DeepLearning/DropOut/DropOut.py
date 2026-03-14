import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from DeepLearning.Softmax.PrepareImageDataset import load_data_fashion_mnist

def plt_show_res(drop_outs,train_losses,test_losses):
    plt.figure(figsize=(8,5))
    plt.plot(drop_outs,train_losses,label="train_loss",color="red",marker="o")
    plt.plot(drop_outs,test_losses,label="test_loss",color="blue",marker="o")
    plt.legend()
    plt.show()
def drop_out_layer(X,drop_out):
    assert 0<=drop_out<=1
    if drop_out==1:
        return 0
    if drop_out==0:
        return X
    mask=torch.rand(size=X.size())
    mask=(mask>drop_out).type(torch.float)
    X=mask*X
    X=X/(1-drop_out)
    return X
# 不推荐X[mask]=0 这种写法 和矩阵乘法相比慢得多
class Net(nn.Module):
    def __init__(self,nums_input,nums_output,nums_hidden1,nums_hidden2,drop_out,is_train=True):
        super(Net,self).__init__()
        self.nums_input=nums_input
        self.nums_output=nums_output
        self.nums_hidden1=nums_hidden1
        self.nums_hidden2=nums_hidden2
        self.drop_out=drop_out
        self.is_train=is_train
        self.lin1=nn.Linear(in_features=nums_input,out_features=nums_hidden1)
        self.lin2=nn.Linear(in_features=nums_hidden1,out_features=nums_hidden2)
        self.lin3=nn.Linear(in_features=nums_hidden2,out_features=nums_output)
        self.relu=nn.ReLU()
    def forward(self,X):
        H1=self.relu(self.lin1(X.reshape(-1,nums_input)))
        if self.is_train:
            H1=drop_out_layer(H1,self.drop_out)
        H2=self.relu(self.lin2(H1))
        if self.is_train:
            H2=drop_out_layer(H2,self.drop_out)
        return self.lin3(H2)
epochs=5
batch_size = 256
n_train=20
n_test=100
lr=0.1
nums_input,nums_output,nums_hidden1,nums_hidden2=784,10,256,256
train_iter,test_iter=load_data_fashion_mnist(batch_size)
loss_fn=nn.CrossEntropyLoss()
drop_outs = []
train_losses = []
test_losses = []
drop_out_range=range(0,10,2)
print("start loop")
for drop_out in drop_out_range:
    drop_out=drop_out/10
    drop_outs.append(drop_out)
    net = Net(nums_input, nums_output, nums_hidden1, nums_hidden2, drop_out)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    for epoch in range(epochs):
        # print(f"epoch {epoch+1} original loss {loss_fn(net(X_train), y_train)}")
        for X,y in train_iter:
            y_hat=net(X)
            loss=loss_fn(y_hat,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(f"epoch {epoch+1} trained loss {loss_fn(net(X_train),y_train)}")
    net.is_train=False
    with torch.no_grad():
        total_train_loss=torch.zeros(1)
        for X,y in train_iter:
            total_train_loss+=loss_fn(net(X),y)
        train_losses.append(total_train_loss.item()/len(train_iter))
        total_test_loss=torch.zeros(1)
        for X,y in test_iter:
            total_test_loss+=loss_fn(net(X),y)
        test_losses.append(total_test_loss.item()/len(test_iter))
    print("finish one loop ")
plt_show_res(drop_outs,train_losses,test_losses)
# 两层模型开始cpu就算的非常慢了 结果
# 通过torch的简洁写法这里就不再细说了 可以像hidden 那样直接添加一个 dropout层就行了 只需要提供一个概率作为参数非常方便
# 这个dropout层会自动根据你的当前状态执行相应操作 简单来说你只需要设置 eval() 和 train() 这两个状态即可
# 最后再提一嘴 很显然 与小模型小丢弃相比 大模型大丢弃的模型一般更优
net=nn.Sequential(nn.Flatten(),nn.Linear(1,2),nn.Dropout(0.5))
