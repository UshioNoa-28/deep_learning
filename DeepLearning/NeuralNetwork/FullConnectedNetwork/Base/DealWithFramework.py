import torch
import torch.nn as nn

from DeepLearning.Softmax.PrepareImageDataset import train_iter

num_input=784
num_output=10
num_hidden=256
lr=0.01
loss=nn.CrossEntropyLoss()

net=nn.Sequential(nn.Flatten(),nn.Linear(num_input,num_hidden),nn.ReLU(),nn.Linear(num_hidden,num_output))
# 继承自nn.Module也行 不过这个方便一点
optimizer=torch.optim.SGD(net.parameters(),lr)
def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,0,0.01)
net.apply(init_weight)
epochs=10
for epoch in range(epochs):
    correct = 0
    total = 0
    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        total+=len(y)
        correct += torch.sum(((y_hat.argmax(axis=1)==y).type(torch.float)))
    print(f"epoch {epoch+1} accurate {correct/total }")
