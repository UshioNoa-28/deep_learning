import torch
import random
def synthetic_data(w,b,num_examples):
    X=torch.normal(0,1,(num_examples,len(w))) # 生成矩阵 数据符合高斯分布 0为均值 1为方差 后面两个参数为矩阵的行列
    w=w.reshape(-1,1) # 不要T转置 而是直接将其转为列向量的形态
    y= X @ w + b
    y+=torch.normal(0,0.01,y.shape)
    return X,y.reshape(-1,1)# -1表示自动计算 即仅保留一列 行你自己算
true_w=torch.tensor([2,-3.4])
true_b=-4.2
features,labels=synthetic_data(true_w,true_b,1000)
def data_iter(batch_size,features,labels):
    num_examples=len(features)# 获取样本数量
    indices=list(range(num_examples))
    random.shuffle(indices)# 就地取随机数
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices] # 每次返回一批数据用来SGD
true_batch_size=10

w=torch.normal(0,1,size=(1,2),requires_grad=True)
b=torch.zeros(size=(1,),requires_grad=True)
def linreg(X,w,b):
    return X@w.T+b #定义模型
def squared_loss(y_hat,y):
    return (y_hat-y)**2/2 # 定义损失函数
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for para in params:
            para-=lr*para.grad/batch_size# grad默认累计
            para.grad.zero_()# 迭代完一次后清零
epochs=3
lr=0.02
net=linreg
loss=squared_loss

for epoch in range(epochs):
    for X,y in data_iter(true_batch_size,features,labels):
        y_hat=net(X,w,b)
        l=loss(y_hat,y)
        l.sum().backward() # 下面细说
        sgd([w,b],lr,true_batch_size)
    with torch.no_grad():
        train_l=loss(net(features,w,b),labels).sum()
        # print(f"epoch {epoch+1} , loss {train_l.item()}")
# 当你创建tensor的时候 可以将 requires_grad设置为true 表示追踪这两个变量的计算
# 接下来的计算都会被pytorch记录为图 比如说上面的y_hat loss的计算 只要是涉及到这两个参数的计算 都会被记录
# 同样sum也是 当调用backward的时候 就会从后往前推通过链式法则一路上将所有的requires_grad的参数的梯度都计算出来 然后将计算的结果添加到参数的.grad中
# with torch.no_grad(): 指的就是关掉梯度追踪 比如说sgd里面的para-=lr*... 此时如果不关掉 那么本次更新也会被记录作为运算
# 同样如果不需要追踪 比如说最后的求损失 为了性能也可以选择将其关掉

