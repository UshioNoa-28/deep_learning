import torch
import torch.nn as nn
def batch_norm(X,gamma,beta,moving_mean,moving_var,eps,momentum):
    # gamma beta 不用多说 moving_mean,moving_var即为全局方差 在做推理的时候时候 可以看onenote
    # eps 为小数字防止输入方差为0的情况 momentum用来作为更新moving_mean 和 moving_var的参数
    if not torch.is_grad_enabled():
        X_hat=(X-moving_mean)/torch.sqrt(moving_var+eps)# 如果为test模式就直接算
    else:
        assert len(X.shape) in (2,4) # 仅支持二维和四维
        if len(X.shape)==2: # 如果是二维的话
            mean=X.mean(dim=0) # 每行是一个数据 求每一个输入的每一维的平均值
            var=X.var(dim=0)
        else:
            # 线性的归一化很简单 对于卷积 具体的操作为
            # 对于一批次的数据 收集该批次中的N个样本所有第c个通道 也就是N*w*h
            # 依照这个步骤得到 一个长度为c的均值向量和方差向量 gamma和beta也是
            # 归一化时 该样本的整个通道的的所有像素 用同一个上面算出来的均值 方差 gamma和beta
            mean=X.mean(dim=(0,2,3),keepdim=True)
            var=X.var(dim=(0,2,3),keepdim=True)
        X_hat=(X-mean)/torch.sqrt(var+eps)
        moving_mean=momentum*moving_mean+(1-momentum)*mean
        moving_var=momentum*moving_var+(1-momentum)*var
    Y=gamma*X_hat+beta
    return Y,moving_mean.data,moving_var.data
# 大概就这样了 简洁实现为



nn.BatchNorm1d(10) # 对于二维维直接给出输入即可
nn.BatchNorm2d(20)# 对于卷积给出通道数即可 上面解释过了只看通道数

# epoch 6 train_loss 0.4748036549143584 test_loss 0.550869483298387
# train accuracy 0.8358125686645508 test accuracy 0.8093050718307495
# epoch 7 train_loss 0.40154175542328346 test_loss 0.615105225588567
# train accuracy 0.8600648045539856 test accuracy 0.8021165728569031
# epoch 8 train_loss 0.3395462996533149 test_loss 0.6000168552985207
# train accuracy 0.8805782198905945 test accuracy 0.8141972422599792
# epoch 9 train_loss 0.2906657525112403 test_loss 0.5821989986081474
# train accuracy 0.8989323377609253 test accuracy 0.8268769979476929
# epoch 10 train_loss 0.23345040936161235 test_loss 0.5943052608745928
# train accuracy 0.918106198310852 test accuracy 0.8290734887123108
# 特地重新跑了一遍AlexNet
# net=nn.Sequential(
#     nn.Conv2d(3,96,kernel_size=11,stride=4,padding=1),
#     nn.BatchNorm2d(96), nn.ReLU(),
#     nn.MaxPool2d(kernel_size=3,stride=2),
#     nn.Conv2d(96,256,kernel_size=5,padding=2),
#     nn.BatchNorm2d(256),nn.ReLU(),
#     nn.MaxPool2d(kernel_size=3,stride=2),
#     nn.Conv2d(256,384,kernel_size=3,padding=1),
#     nn.BatchNorm2d(384),nn.ReLU(),
#     nn.Conv2d(384,384,kernel_size=3,padding=1),
#     nn.BatchNorm2d(384),nn.ReLU(),
#     nn.Conv2d(384,256,kernel_size=3,padding=1),
#     nn.BatchNorm2d(256),nn.ReLU(),
#     nn.MaxPool2d(kernel_size=3,stride=2),
#     nn.Flatten(),
#     nn.Linear(6400,4096),
#     nn.BatchNorm1d(4096),nn.ReLU(),nn.Dropout(0.5),
#     nn.Linear(4096,4096),
#     nn.BatchNorm1d(4096),nn.ReLU(),nn.Dropout(0.5),
#     nn.Linear(4096,10)
# )
