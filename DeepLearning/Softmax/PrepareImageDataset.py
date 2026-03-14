import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import time
trans=transforms.ToTensor() # 相当于做一个变换 简单来说就是将PIL类型转为tensor类型 操作包括将 H W 转为 C H W 同时将像素归一化 归一化也是为了更好地训练
# 也就是将所有的像素值/255变为0~1.0之间
mnist_train=torchvision.datasets.FashionMNIST(root="./Data",train=True,transform=trans,download=True)
mnist_test=torchvision.datasets.FashionMNIST(root="./Data",train=False,transform=trans,download=True)
# 需要注意的是以上的下载操作仍然下载的是PIL数据 只不过你每次往里面取数据的时候会自动帮你执行该transform
# print(len(mnist_train))
# print(len(mnist_test))

# print(mnist_train[0][0].shape) # [0][0]表示第0个example的第0张图片

batch_size=256
def get_data_load_worker():
    return 0
train_iter=data.DataLoader(mnist_train,num_workers=get_data_load_worker(),batch_size=batch_size,shuffle=True)
# num_work为读取数据的进程数量 司马window强制要求放到main函数里面 所以这里设置为0 linux听说没有类似问题
s=time.time()
for X,y in train_iter:
    continue
e=time.time()
# print(f"end up with {e-s} s")

# 最后打包成一个函数即可
def load_data_fashion_mnist(batch_size):
    trans=transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root="./Data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./Data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,num_workers=get_data_load_worker()),
            data.DataLoader(mnist_test,batch_size=batch_size,shuffle=True,num_workers=get_data_load_worker()))
# DataLoader 对batch_size的实现是将batch_size个向量抽出来 然后将他们拼成一个大的tensor