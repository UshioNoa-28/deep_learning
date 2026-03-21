from torch import nn

class SimpleAdapter(nn.Module):
    """
    简单的一个Adapter 内部仅包含线性变化和一个LeakyRelu激活函数
    """
    def __init__(self, c_in, c_out=768):
        super(SimpleAdapter, self).__init__()
        self.fc = nn.Sequential(nn.Linear(c_in, c_out, bias=False), nn.LeakyReLU())
    def forward(self, x):
        x = self.fc(x)
        return x
class SimpleProj(nn.Module):
    """
    简单的一个可训练的Project 同样线性变化加个LeakyRelu
    """
    def __init__(self, c_in, c_out=768, relu=True):
        super(SimpleProj, self).__init__()
        if relu:
            self.fc = nn.Sequential(nn.Linear(c_in, c_out, bias=False), nn.LeakyReLU())
        else:
            self.fc = nn.Linear(c_in, c_out, bias=False)
    def forward(self, x):
        x = self.fc(x)
        return x