import torch
import torch.nn as nn
class Decoder(nn.Module):
    def __init__(self,N,embed_dim,vocab_size,nums_heads,fc_size,max_seq_len):
        super().__init__()
        self.N=N
        self.embed_fn=nn.Embedding(vocab_size,embed_dim)
        self.position=nn.Embedding(max_seq_len,embed_dim)
        self.attention=nn.MultiheadAttention(embed_dim,nums_heads,batch_first=True,dropout=0.5)
        self.cross_attention=nn.MultiheadAttention(embed_dim,nums_heads,batch_first=True,dropout=0.5)
        self.norm1=nn.LayerNorm(embed_dim)
        self.norm2=nn.LayerNorm(embed_dim)
        self.norm3=nn.LayerNorm(embed_dim)
        self.ff=nn.Sequential(
            nn.Linear(embed_dim,fc_size),
            nn.ReLU(),
            nn.Linear(fc_size,embed_dim)
        )
    def forward(self,text,encoder_output):
        text=self.embed_fn(text)
        seq_len=text.size(1)# [batch_size,seq_len,embed_dim]
        positions=self.position(torch.arange(0,seq_len,device=text.device))
        text+=positions
        mask=torch.triu(torch.full((seq_len,seq_len),float("-inf")),1) # full 是创建一个全为某个值的矩阵
        # triu取上三角矩阵diagonal 如果是1表示去掉对角线 0 表示包含对角线 我们想要去掉对角线
        for i in range(self.N):
            y,_=self.attention(text,text,text,attn_mask=mask)
            text=self.norm1(text+y)
            y,_=self.cross_attention(text,encoder_output,encoder_output)
            text=self.norm2(text+y)
            y=self.ff(text)
            text=self.norm3(y+text)
        return text