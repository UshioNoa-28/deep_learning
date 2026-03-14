import torch
import torch.nn as nn
def nin_block(in_channels,out_channels,kernal_size,padding=0,stride=1):
    net=nn.Sequential(
    nn.Conv2d(in_channels,out_channels,kernel_size=kernal_size,padding=padding,stride=stride),
    nn.ReLU(),
    nn.Conv2d(out_channels,out_channels,kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(out_channels,out_channels,kernel_size=1),
    nn.ReLU(),
)

