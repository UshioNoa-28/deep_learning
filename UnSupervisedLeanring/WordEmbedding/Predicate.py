# 由于torchtext版本对不上 所以直接手动下载文本数据放到Data里面了
from collections import Counter
import torch
import torch.nn as nn
import torch.utils.data as data

with open("data/train.txt","r",encoding="utf8") as f:
    text = f.read()
tokens = text.lower().split() # 小写 按空格分开
counter = Counter(tokens) # 统计词频 counter=Counter(tokens) 结果为类似于 {"cat":100,"dog":200}这样的结果
vocab_size = 2000 # 词表大小
dim=128 # Embedding的NeuroNetwork的线性层的输入
vocab = {w:i for i,(w,_) in enumerate(counter.most_common(vocab_size))} # 构建词表most_common会返回上面的字典中最大数量的词
# 然后构造 w:i w即为后面便利的字典的word i即为id 最后得到的结果为 {"the":0...}为每一个词提供一个Id
idx_to_word = {i: w for w, i in vocab.items()} # 构建一个反向词典 等会通过Id来找word
ids = [vocab.get(w,vocab["<unk>"]) for w in tokens] # 对于token中的所有词 vocab如果有那就返回w的Id 没有那就返回vocab["<unk>"]的Id
# 也就是将文章编码为类似于[1,2,2000,12,]这样的形式
window = 1 # window表示上下文窗口大小
contexts=[]
targets=[]
for i in range(window,len(ids)-window):
    context=[ids[i-1],ids[i+1]] # 获取上下文
    target=ids[i] # 获取目标
    contexts.append(context)
    targets.append(target)
    # 然后将上下文和目标存到列表中
class CBOW(nn.Module):
    def __init__(self,vocab_size,embedding_size):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,embedding_size)
        # Embedding层 让你告别one-hot向量 也就是说你不再需要手动转为one-hot向量 可以直接输入词的Id 因为这样做非常浪费内存和计算资源
        # 输入两个 第一个为词的维度n 第二个为你要的权重的数量m 他就会随机初始化一个n*m的矩阵
        # Embedding层根据你的输入 她不会做任何乘法 而是类似于hash表 比如说你给一个单词的Id
        # torch.tensor([5]) 输入给他时 不会做任何矩阵乘法 而是直接去查询第5行整个权重拿给你 也就是得到一个 1*128的矩阵然后再经过线性变换得到结果
        self.l=nn.Linear(embedding_size,vocab_size)
    def forward(self,X):
        return self.l(self.embed(X).mean(dim=1))
epochs=3
lr=0.02
batch_size=256
net=CBOW(len(vocab),embedding_size=dim)
net=net.to("cuda")
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(lr=lr,params=net.parameters())
train_dataset=data.TensorDataset(torch.tensor(contexts),torch.tensor(targets))
train_iter=data.DataLoader(train_dataset,shuffle=True,batch_size=batch_size)

for epoch in range(epochs):
    for X,y in train_iter:
        X,y=X.to("cuda"),y.to("cuda")
        y_hat=net(X)
        optimizer.zero_grad()
        loss=loss_fn(y_hat,y)
        loss.backward()
        optimizer.step()
while True:
    w1=input("word1:")
    w2=input("word2:")
    x=torch.tensor([[vocab.get(w1,vocab["<unk>"]),vocab.get(w2,vocab["<unk>"])]])
    x.to("cuda")
    y=net(x)
    pred=y.argmax().item()
    word = idx_to_word.get(pred, "<unk>")
    print("prediction:",word)
    # epoch 1 loss 4.390010833740234
    # epoch 6 loss 4.310044765472412
    # epoch 11 loss 3.3617701530456543
    # epoch 16 loss 2.8919405937194824
    # word1: barker
    # word2: born
    # prediction: was
    # word1: cat
    # word2: sofat
    # prediction: < unk >
    # word1: the
    # word2: of
    # prediction: < unk >
    # word1: until
    # word2: 1919
    # prediction: < unk >
    # word1: this
    # word2: my
    # prediction: < unk >
    # word1: what
    # word2: you
    # prediction: know
