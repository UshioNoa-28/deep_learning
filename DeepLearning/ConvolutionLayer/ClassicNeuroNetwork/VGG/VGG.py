import torch
import torch.nn as nn
def vgg_block(num_block,in_channels,out_channels):
    layers=[]
    for _ in range(num_block):
        layers.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        layers.append(nn.ReLU())
        in_channels=out_channels
    layers.append(nn.MaxPool2d(kernel_size=3,stride=2))
    return nn.Sequential(*layers)
# VGG