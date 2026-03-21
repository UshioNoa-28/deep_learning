# pytorch自然也是对Attention层做了支持 multi_header也被支持了
import torch.nn as nn
import torch
class SelfAttentionModule(nn.Module):
    def __init__(self,embedding_dim,num_headers,fc1,num_states,vocab_size,max_seq_len):
        super().__init__()
        self.attention1=nn.MultiheadAttention(embed_dim=embedding_dim,num_heads=num_headers,dropout=0.5,batch_first=True)
        # Batch_first 表示输入为[batch_size,seq_len,embed_dim]
        # 假设输入为 one two three 经过 word embedding 得到 1 * embed_len的向量
        # 也就是说 seq_len为3 embed_dim为 embed_len
        # 他的WqWkWv三个权重矩阵形状为 embed_len * embed_len 得到的 QKV即为seq_len*embed_len
        # 然后相关性即为 Q*K^t 即 seq_len*seq_len 层数搞了 就跟复杂 主要是为了加速运算会涉及到矩阵的拼接什么的
        # 两个输出attn_output 经过注意力加权之后的特征 也就是Attention层的结果 [batch_size,seq_len]
        # attn_weights 每个词对其他词的注意力权重分数 结果大小为 [batch_size,seq_len,seq_len]
        # 需要注意的是 要求embed_dim%num_headers=0 主要是pytorch底层做了点优化 顺从就行
        self.attention2=nn.MultiheadAttention(embed_dim=embedding_dim,num_heads=num_headers,dropout=0.5,batch_first=True)
        self.embed=nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
        self.l1=nn.Linear(embedding_dim,fc1)
        self.norm = nn.LayerNorm(fc1)
        self.l2=nn.Linear(fc1,num_states)
        self.position=nn.Embedding(max_seq_len,embedding_dim)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.5)
        # 用embedding和one-hot向量表示位置
    def forward(self,text):
        seq_len=text.size(1) # 用来做位置编码 大小即为序列的长度和每个词的word embedding维度
        positions=torch.tensor(torch.arange(0,seq_len),device=text.device)
        text=self.embed(text)+self.position(positions)# 加个位置编码
        text,_=self.attention1(text,text,text)
        text,_=self.attention2(text,text,text) # attention的forward函数接收三个参数
        # 就是你要求QKV的时候 矩阵*的东西 这里简单的attention就都是自己了 主要是为了支持 cross_attention
        # 别忘了还要压平 原来的大小是[batch_size,seq_len,embed_dim]
        # 由于每个序列可能不同 这里用之前的Flatten的话 我们无法确定l1的in_feature 所以
        # text=text.mean(dim=1) # 按序列这个维度 直接求一个句子的每个词的输出向量的平均
        # 或者下面这个做法 直接取第一个词的输出
        text=text[:,0,:]# 取每个序列的第一个向量 一般而言上面那个可能更推荐
        text=self.relu(self.norm(self.l1(text)))
        text=self.dropout(text)
        text=self.l2(text)
        return text
# epoch 15 test loss 0.49467376300266813 train loss 0.1780409961938858 accuracy 0.8454440236091614
# epoch 16 test loss 0.5170724881546838 train loss 0.17337875068187714 accuracy 0.8330449461936951
# epoch 17 test loss 0.46963689582688467 train loss 0.16523785889148712 accuracy 0.855247974395752
# epoch 18 test loss 0.4921324976852962 train loss 0.14890965819358826 accuracy 0.855247974395752
# epoch 19 test loss 0.555512432541166 train loss 0.13063739240169525 accuracy 0.8497692942619324
# epoch 20 test loss 0.5253045431205204 train loss 0.12348521500825882 accuracy 0.8483275175094604
# Attention作为一个Transformer的铺垫 性能还没有LSTM高 等我Transformer杀回来:(