import torch
import torch.nn as nn
echos=1000000
X=torch.tensor([[1,1],[1,0],[0,1],[0,0]],dtype=torch.float)
y=torch.tensor([[0],[1],[1],[0]],dtype=torch.float)
class XorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1=nn.Linear(in_features=2,out_features=5)
        self.L2=nn.Linear(in_features=5,out_features=1)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.sigmoid(self.L2(self.relu(self.L1(x))))
    pass
model_0=XorModel()
Loss_fn=torch.nn.MSELoss()

Optimizer=torch.optim.SGD(params=model_0.parameters(),lr=0.02)
print("Initial")
print(model_0.state_dict())
for echo in range(echos):
    model_0.train()
    y_preds=model_0(X)
    loss=Loss_fn(y_preds,y)
    Optimizer.zero_grad()
    loss.backward()
    Optimizer.step()
print("After")
print(model_0.state_dict())
X_test=torch.tensor([[1,0],[0,1],[1,1],[0,0]],dtype=torch.float)
y_test=torch.tensor([[1],[1],[0],[0]],dtype=torch.float)
y_pred=model_0(X_test)
print(y_pred)

