import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from GetData import get_data


class Reshape(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,X):
        return X.reshape((-1,1,32,32))
torch.manual_seed(42)
net=nn.Sequential(
    Reshape(),
    nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Flatten(),
    nn.Linear(in_features=16*5*5,out_features=120),
    nn.ReLU(),
    nn.Linear(in_features=120,out_features=84),
    nn.ReLU(),
    nn.Linear(in_features=84,out_features=10)
)

def draw_image(epochs,train_losses,test_losses):
    plt.figure(figsize=(8,5))
    plt.plot(epochs,train_losses,label="train",color="blue",marker="o")
    plt.plot(epochs,test_losses,label="test",color="red",marker="o")
    plt.xlabel="epoch"
    plt.ylabel="loss"
    plt.legend()
    plt.show()
def get_accuracy(y_hat,y):
    return (y_hat.argmax(axis=1)==y).type(torch.float).sum().item()/len(y_hat)

epochs=10
lr=0.05
batch_size=256
weight_decay=0.01
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=lr,weight_decay=weight_decay)
train_iter,test_iter=get_data(batch_size)
train_losses=[]
test_losses=[]
for epoch in range(epochs):
    train_loss,test_loss=0,0
    train_accuracy,test_accuracy=0,0
    net.train()
    for X,y in train_iter:
        y_hat=net(X)
        optimizer.zero_grad()
        l=loss_fn(y_hat,y)
        train_loss += l.item()
        l.backward()
        optimizer.step()
        train_accuracy+=get_accuracy(y_hat,y)
    net.eval()
    with torch.no_grad():
        for X,y in test_iter:
            y_hat=net(X)
            test_loss+=loss_fn(y_hat,y).item()
            test_accuracy+=get_accuracy(y_hat,y)
        train_loss/=len(train_iter)
        test_loss/=len(test_iter)
        train_accuracy/=len(train_iter)
        test_accuracy/=len(test_iter)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"epoch {epoch+1} : ")
        print(f"train loss {train_loss} test_loss {test_loss}")
        print(f"train accuracy {train_accuracy} test accuracy {test_accuracy}")
draw_image(range(epochs), train_losses, test_losses)
torch.save(net.state_dict(), "LeNet")





