import torch
import torch.nn as nn

from DeepLearning.Softmax.PrepareImageDataset import load_data_fashion_mnist

batch_size=256
train_iter,test_iter=load_data_fashion_mnist(batch_size)
def relu(X:torch.Tensor):
    a=torch.zeros_like(X)
    return torch.max(X,a)
num_input=784
num_output=10
num_hidden=256
lr=0.01
W1=nn.Parameter(torch.randn(num_input,num_hidden),requires_grad=True)
b1=nn.Parameter(torch.zeros(num_hidden),requires_grad=True)
W2=nn.Parameter(torch.randn(num_hidden,num_output),requires_grad=True)
b2=nn.Parameter(torch.zeros(num_output),requires_grad=True)
paras=[W1,b1,W2,b2]
softmax=nn.Softmax()
def net(X):
    X=X.reshape(-1,num_input)
    return (relu(X@W1+b1))@W2+b2
def accuracy(y_hat,y):
    y_hat=y_hat.argmax(axis=1) # 寻找最大值的位置 axis=0 表示沿着列查找最大值出现的位置 axis=1表示沿着行查找最大值出现的位置
    cmp=y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())
optimizer=torch.optim.SGD(paras,lr)
loss=nn.CrossEntropyLoss()
epochs=10
for epoch in range(epochs):
    correct=0
    total=0
    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        correct+=accuracy(y_hat,y)
        total+=len(y)
    print(f"epoch {epoch+1} accuracy {correct/total}")
total=0
correct=0
for X,y in test_iter:
    total+=len(y)
    correct+=accuracy(net(X),y)
print(f"test accurate {correct/total}")
# 输出
# epoch 1 accuracy 0.54805
# epoch 2 accuracy 0.6931666666666667
# epoch 3 accuracy 0.7268833333333333
# epoch 4 accuracy 0.7425
# epoch 5 accuracy 0.7535
# epoch 6 accuracy 0.7632
# epoch 7 accuracy 0.7691666666666667
# epoch 8 accuracy 0.7741666666666667
# epoch 9 accuracy 0.7777333333333334
# epoch 10 accuracy 0.7806
# test accurate 0.7748