# 微调 来试试水 而pytorch刚好提供了非常多的预训练好的模型可以直接使用
import torchvision.models as models
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
model=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # 加载在ImageNet-1K上预训练好的模型
# 然后你就可以直接使用了只需要加个分类头再稍微训练一下即可
transform=transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])
# 由于残差太现代了 而且还是从ImageNet下的 3*224*224
mnist_train = torchvision.datasets.FashionMNIST(
    root="./Data",      # 数据存放目录
    train=True,         # 训练集
    download=False,      # 自动下载
    transform=transform
)
mnist_test = torchvision.datasets.FashionMNIST(
    root="./Data",
    train=False,
    download=False,
    transform=transform
)


batch_size=256
train_iter=data.DataLoader(mnist_train,batch_size,shuffle=True)
test_iter=data.DataLoader(mnist_test,shuffle=False)
num_classes=10
model.fc=nn.Linear(model.fc.in_features,num_classes)

classify_parameters=[]
feature_extractors=[]
for name,parameter in model.named_parameters():
    if "fc" in name:
        classify_parameters.append(parameter)
    else:
        feature_extractors.append(parameter)
# 分两类参数 分别设置不同的学习率
parameter_group=[
    {"params":feature_extractors,"lr":1e-4,"name":"features"},
    {"params":classify_parameters,"lr":1e-2,"name":"classifier"}
]
optimizer=optim.SGD(params=parameter_group,lr=1e-2)# 这里设置的学习率是默认值 实际上会被组内设置的lr覆盖
loss_fn=nn.CrossEntropyLoss()
epochs=3
model=model.to("cuda")
for epoch in range(epochs):
    train_loss,test_loss=0,0
    model.train()
    for X,y in train_iter:
        X,y=X.to("cuda"),y.to("cuda")
        y_hat=model(X)
        optimizer.zero_grad()
        loss=loss_fn(y_hat,y)
        loss.backward()
        optimizer.step()
        train_loss+=loss
    model.eval()
    with torch.no_grad():
        for X,y in test_iter:
            X, y = X.to("cuda"), y.to("cuda")
            y_hat=model(X)
            loss=loss_fn(y_hat,y)
            test_loss+=loss
        train_loss/=len(train_iter)
        test_loss/=len(test_iter)
    print(f"epoch {epoch+1} train loss {train_loss} test loss {test_loss}")
