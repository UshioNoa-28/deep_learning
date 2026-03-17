# 自编码器 简单来说 就是用尽可能少的维数表达相对较多的信息
# 分为三部分 编码器 瓶颈 解码器
# 自编码器不需要任何监督学习 因为这本身就不需要标签 你只需要搞个网络 压缩 解压 然后损失函数定义为 解压和压缩前的文件的差异即可
# 接下来就是求导 反向传播 就能正常训练了 然后 只需要带着编码器 因为它非常善于提取特征 对于新的数据集 略加训练即可
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
transform=transforms.Compose([
    transforms.ToTensor()
])
train_dataset=torchvision.datasets.MNIST(root="./data",download=True,transform=transform,train=True)
test_dataset=torchvision.datasets.MNIST(root="./data",download=True,transform=transform,train=False)
batch_size=256
lr=0.1
encode_dim=32
epochs=20
train_iter=data.DataLoader(train_dataset,shuffle=True,batch_size=batch_size,num_workers=4)
test_iter=data.DataLoader(test_dataset,shuffle=False,batch_size=batch_size,num_workers=4)
# 本质上也是一个很简单的东西 对其降维即可
loss_fn=nn.MSELoss()
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Linear(28*28,256),nn.ReLU(),
            nn.Linear(256,128),nn.ReLU(),
            nn.Linear(128,encode_dim)
        )
        self.decoder=nn.Sequential(
            nn.Linear(encode_dim,128),nn.ReLU(),
            nn.Linear(128,256),nn.ReLU(),
            nn.Linear(256,28*28),
            nn.Sigmoid()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed
net=AutoEncoder()
net=net.to("cuda")
optimizer=torch.optim.SGD(net.parameters(),lr=lr)
for epoch in range(epochs):
    train_loss, test_loss = 0, 0
    net.train()
    for X,_ in train_iter:
        X=X.reshape(-1,28*28)
        X=X.to("cuda")
        y_hat=net(X)
        optimizer.zero_grad()
        loss=loss_fn(y_hat,X)
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    net.eval()
    for X,_ in test_iter:
        with torch.no_grad():
            X = X.reshape(-1, 28 * 28)
            X=X.to("cuda")
            y_hat=net(X)
            loss=loss_fn(y_hat,X).item()
            test_loss+=loss

# 从测试集中取一些图片
with torch.no_grad():  # 在这个代码块中，我们不需要计算梯度
    data_iter = iter(test_iter)
    images, _ = next(data_iter)

    # 将图片展平输入模型
    images_flat = images.view(images.size(0), -1).to("cuda")
    reconstructed_flat = net(images_flat)
    # 将重建的向量恢复成图片形状
    reconstructed = reconstructed_flat.cpu().view(reconstructed_flat.size(0), 1, 28, 28)

    # 可视化对比
    n_images = 10
    plt.figure(figsize=(20, 4))
    for i in range(n_images):
        # 显示原始图片
        ax = plt.subplot(2, n_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n_images // 2:
            ax.set_title('Original Images')

        # 显示重建图片
        ax = plt.subplot(2, n_images, i + 1 + n_images)
        plt.imshow(reconstructed[i].squeeze(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n_images // 2:
            ax.set_title('Reconstructed Images')
    plt.show()


