import torch
import torch.nn as nn
# 继承自模型基于最高的自由度  sequential更加方便 下面给出模仿的实现
class My_Sequential(nn.Module):
    def __init__(self,*args):
        super().__init__()
        for block in args:
            self._modules[block]=block# 这个module 当你调用self.a=value的时候
            # 就会判断value本身是否是Module的子类 如果是那就将其加入_module字典 也就是说实际上你在init里面的涉及模型的赋值
            # 都会被添加到该字典中 如果是一个 nn.Parameter对象 则会被添加到_parameters里面
            # 上面的两个主要是起存储作用 简单来说就是 当你访问例如 Parameter 或者直接print(net)的时候会将信息展示给你
            # 你也可以通过 net["key"] 来访问特定层的module sequence 默认key为0 1 2
    def forward(self,X):
        for block in self._modules.values():
            X=block(X)
            return X
# 这样就是一个自定义的Sequential
# 一般而言 层基本上指的是一次矩阵运算 块则是多层组合而成的功能组件 简单来说就是对层的抽象 比如说Module
# 里面包含多个层 你也不需要关心里面是什么 把他当成一个层用即可 nn.Linear()即为一个Module 只不过仅包含一个线性变化层而已
net=nn.Sequential(nn.Linear(2,4))
print(net[0].state_dict())# 这样即可访问该module的_module
print(type(net[0].bias)) # <class 'torch.nn.parameter.Parameter'>
print(net[0].bias)# tensor([ 0.1794, -0.6224,  0.2399,  0.2971], requires_grad=True) 偏置即为nn.Parameter
# parameter 可以通过data访问数据和grad访问梯度
print(net)
