import torch
import torch.nn as nn
def pooling(X,pool_size,mode="max"):
    p_h,p_w=pool_size
    Y=torch.zeros(size=(X.shape[0]-p_h+1,X.shape[1]-p_w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode=="max":
                Y[i,j]=torch.max(X[i:i+p_h,j:j+p_w])
            if mode=="mean":
                Y[i,j]=(X[i:i+p_h,j:j+p_w].type(torch.float)).mean()
    # 注意访问最好通过[i,j]访问 性能高且符合直接 两个索引不太符合直觉
    return Y
X=torch.tensor([
    [1,2,3],
    [3,4,1],
    [1,7,10]
])
print(pooling(X,(2,2)))
print(pooling(X,(2,2),"mean"))
Y=torch.tensor([
    [1,2,3],
    [1,2,3],
    [1,2,3]
])
Y=Y.reshape((1,1)+Y.shape)# 要搞成四维或者三维
# 深度学习框架中步幅默认和池化窗口大小相同
pool2d=nn.MaxPool2d(2,stride=1) # 当然你可以手动设置 和之前按的卷积的步长和填充一样
# nn.AvgPool2d 平均数版本也有提供

print(pool2d(Y))