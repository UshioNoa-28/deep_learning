import torch
x=torch.rand(size=(2,))
x_reshape=x.reshape(1,2)
print(x)
print(x_reshape)
# 仅允许不损失信息的改变形状 比如说添加一个维度 从 2 变成 1 2

# 还可以让堆叠张量
x_stack=torch.stack([x,x,x,x],dim=1)
print(x_stack)