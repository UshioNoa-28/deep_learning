import torch.nn as nn
hidden_layer=128
class LSTMModule(nn.Module):
    # 事实上 LSTM和RNN的用法完全一致 除了返回值以外 知道输入维度和内存大小就能退出三个门的向量 不信自己试试
    # 返回值由out_put,h_n 变为(h_n,c_n) 其他不变 c_n新东西 实际上也就是最后一个时刻 每个层的MemoryCell的C
    # 实际上还是h_n用的多 主要是
    def __init__(self,vocab_size,embedding_dim,hidden_size,nums_layer,state_nums):
        super().__init__()
        self.hidden_size=hidden_size
        self.nums_layer=nums_layer
        self.embedding=nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
        self.rnn=nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,num_layers=nums_layer,batch_first=True,dropout=0.5)
        self.fc1=nn.Sequential(nn.Linear(in_features=hidden_size*nums_layer,out_features=hidden_layer),
                               nn.BatchNorm1d(hidden_layer),nn.ReLU())
        self.fc2=nn.Linear(in_features=hidden_layer,out_features=state_nums)
    def forward(self,text):
        _,(h_n,c_n)=self.rnn(self.embedding(text))
        h_n=h_n.permute(1,0,2)
        h_n=h_n.reshape((-1,self.hidden_size*self.nums_layer))
        return self.fc2(self.fc1(h_n))
