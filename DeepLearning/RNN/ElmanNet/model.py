import torch
import torch.nn as nn
# input_size为每个词的输入维度 比如说可以直接one-hot 2000维 当然为了防止计算量过大 一般是在embedding之后
# hidden_size 为隐藏层的大小 也就是内存能存的数据的量
# num_layer即为RNN的层数
# 输出有两个 output 和 h_n
# output为RNN在最后一层每个时间步计算出来的隐藏状态 h_t batch_first参数指的是是否要将batch放到第一维 主要是好看 习惯也是放到第一维 建议开启
# 形状为 (batch, seq_len, hidden_size) 抛开batch不谈 也就是对于seq里面的每个词 都提供最后算出来的激活值 激活值的形状为hidden_size
# 语音翻译 slot filling 需要关心每个词的需要这个
# h_n 则简短的多
# (num_layers, batch, hidden_size) 仅包含最后一个时间步的隐藏状态 如果有多层 那就是每一层的隐藏状态
# 情感判断 整个句子作为一个整体关心结果
# 往往为了简洁 我们会只取最后一个layer的结果
hidden_layer=128
class RnnModule(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_size,nums_layer,state_nums):
        super().__init__()
        self.hidden_size=hidden_size
        self.nums_layer=nums_layer
        self.embedding=nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
        self.rnn=nn.RNN(input_size=embedding_dim,hidden_size=hidden_size,num_layers=nums_layer,batch_first=True,dropout=0.5)
        self.fc1=nn.Sequential(nn.Linear(in_features=hidden_size*nums_layer,out_features=hidden_layer),
                               nn.BatchNorm1d(hidden_layer),nn.ReLU())
        self.fc2=nn.Linear(in_features=hidden_layer,out_features=state_nums)
        # 这里我们选择仅关注整个过程的最后一个时刻的最后的layer层的隐藏状态 feature_in是hidden_size*nums_layer
    def forward(self,text):
        _,h_n=self.rnn(self.embedding(text))
        h_n=h_n.permute(1,0,2)
        h_n=h_n.reshape((-1,self.hidden_size*self.nums_layer))
        return self.fc2(self.fc1(h_n))