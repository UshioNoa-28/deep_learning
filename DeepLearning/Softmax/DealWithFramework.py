import torch.optim

from PrepareImageDataset import load_data_fashion_mnist
import torch.nn as nn
batch_size=256
num_input=784
num_output=10
lr=0.01
train_iter,test_iter=load_data_fashion_mnist(batch_size)
net=nn.Sequential(nn.Flatten(),nn.Linear(in_features=num_input,out_features=num_output))
# flatten 用于展平 简单来说就是 将第一个维度保留 其后所有的维度转为一维 最终得到一个2d的张量 在这里就会转为矩阵 batch_size*784
def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,0,0.01)
        nn.init.normal_(m.bias,0,0.01)
net.apply(init_weight) # apply会遍历所有的模块及其子模块调用传入的函数 这里net本身即为module 他的子模块 flatten和linear也会被调用
loss=nn.CrossEntropyLoss()# 要求输入为未经过softmax的原始输出如[1,2.3,1.91] 标签为类别索引 会自动帮你处理softmax
optimizer=torch.optim.SGD(net.parameters(),lr=lr)
epochs=10
for epoch in range(epochs):
    total=0.0
    accuracy=0.0
    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        cmp=(y_hat.argmax(axis=1)==y).type(torch.float)
        total+=len(y)
        accuracy+=cmp.sum()
    print(f"epoch {epoch+1} accurate {accuracy/total}")
total=0
correct=0
for X,y in test_iter:
    total+=len(y)
    correct+=accuracy(net(X),y)
print(f"test accurate {correct/total}")
# 结果如下
# epoch 1 accuracy 0.5600166666666667
# epoch 2 accuracy 0.6925333333333333
# epoch 3 accuracy 0.7253666666666667
# epoch 4 accuracy 0.7420666666666667
# epoch 5 accuracy 0.7541666666666667
# epoch 6 accuracy 0.76205
# epoch 7 accuracy 0.7685333333333333
# epoch 8 accuracy 0.7746
# epoch 9 accuracy 0.7786833333333333
# epoch 10 accuracy 0.784
# test accurate 0.7684