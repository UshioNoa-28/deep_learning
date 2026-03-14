import torch
import torch.nn as nn

tensor1=torch.tensor([1,23,4])
torch.save(tensor1, "tensor_file")

tensor2=torch.load("tensor_file")
print(tensor2)
# 这样即可保存一个张量 本质上存一个模型就是存参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self._modules["0"]=nn.Linear(3,1)
    def __getitem__(self, idx):
        return list(self._modules.values())[idx]
    def forward(self,X):
        return self._modules["0"].value()(X)
net=MLP()
def init_weight(m):
    if type(m)==nn.Linear:
        nn.init.uniform_(m.weight,-10,10)
torch.save(net.state_dict(), "MLP")
# 读取
clone=MLP()
clone.load_state_dict(torch.load("MLP"))
print(clone[0].weight==net[0].weight) # tensor([[True, True, True]])