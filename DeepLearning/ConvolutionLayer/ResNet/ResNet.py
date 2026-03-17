import torch
import torch.nn as nn
class Residual(nn.Module):
    def __init__(self,in_channels,out_channels,use_1x1conv=False,stride=1):
        super().__init__()
        self.Conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=stride)
        self.Conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        if use_1x1conv:
            self.Conv3=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
        else:
            self.Conv3=None
        self.relu=nn.ReLU()
        self.Bn1=nn.BatchNorm2d(out_channels)
        self.Bn2=nn.BatchNorm2d(out_channels)
    def forward(self,X):
        Y=self.relu(self.Bn1(self.Conv1(X)))
        Y=self.Bn2(self.Conv2(Y))
        if self.Conv3:
            X=self.Conv3(X)
        return self.relu(Y+X)
X=torch.randn(size=(1,3,4,4))
rbl=Residual(X.shape[1],8,True)
X=rbl(X)
print(X.shape)