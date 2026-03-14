# 实际上为加强版的LeNet 主要体现在卷积层数量增加
# 激活函数变为Relu 使用dropout和数据增强 这个cpu已经跑不动了 所以不要在这里运行
import torch
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_cifar10_loaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )

    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


net=nn.Sequential(
    nn.Conv2d(3,96,kernel_size=11,stride=4,padding=1),
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
net.to("cuda")
def get_accuracy(y_hat,y):
    return (y_hat.argmax(axis=1)==y).type(torch.float).sum()/len(y_hat)

train_iter,test_iter=get_cifar10_loaders(batch_size=256)
loss_fn=nn.CrossEntropyLoss()
lr=0.01
optimizer=torch.optim.SGD(lr=lr,params=net.parameters())
epochs=100
for epoch in range(epochs):
    train_loss,test_loss=0,0
    train_accuracy,test_accuracy=0,0
    net.train()
    for X,y in train_iter:
        X,y=X.to("cuda"),y.to("cuda")
        y_hat=net(X)
        optimizer.zero_grad()
        loss=loss_fn(y_hat,y)
        loss.backward()
        optimizer.step()
        train_loss+=loss
        train_accuracy+=get_accuracy(y_hat, y)
    net.eval()
    for X,y in test_iter:
        X,y=X.to("cuda"),y.to("cuda")
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        test_loss += loss
        test_accuracy+=get_accuracy(y_hat,y)
    train_loss/=len(train_iter)
    test_loss/=len(test_iter)
    train_accuracy/=len(train_iter)
    test_accuracy/=len(test_iter)
    print(f"epoch {epoch + 1} train_loss {train_loss} test_loss {test_loss}")
    print(f"train accuracy {train_accuracy} test accuracy {test_accuracy}")
# epoch 6
# train_loss 1.4511062587520211
# test_loss 1.31263216844382
# train accuracy 0.4700895845890045
# test accuracy 0.5236621499061584
# epoch 7
# train_loss 1.3147627951926477
# test_loss 1.4172760331973482
# train accuracy 0.5246121287345886
# test accuracy 0.513877809047699
# epoch 8
# train_loss 1.194813017080933
# test_loss 1.2428538972577348
# train accuracy 0.5722368955612183
# test accuracy 0.565994381904602
# epoch 9
# train_loss 1.0858471761204382
# test_loss 1.0652729760343655
# train accuracy 0.6163627505302429
# test accuracy 0.6341853141784668
# epoch 10
# train_loss 0.9848676518568684
# test_loss 0.9867959067273063
# train accuracy 0.6523512601852417
# test accuracy 0.6543530225753784