import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,embed_dim,N,fc_size,vocab_size,max_seq_len,num_heads):
        super().__init__()
        self.N=N
        self.embed_dim=embed_dim
        self.norm1=nn.LayerNorm(embed_dim)
        self.norm2=nn.LayerNorm(embed_dim)
        self.ff=nn.Sequential(
            nn.Linear(embed_dim,fc_size),
            nn.ReLU(),
            nn.Linear(fc_size,embed_dim)
        )
        self.embed_fn=nn.Embedding(vocab_size,embed_dim)
        self.attention=nn.MultiheadAttention(embed_dim,num_heads,batch_first=True)
        self.position=nn.Embedding(max_seq_len,embed_dim)
    def forward(self,text):
        seq_len=text.size(1)
        text=self.embed_fn(text)
        positions = torch.arange(seq_len, device=text.device).unsqueeze(0)
        position_embed = self.position(positions)
        text += position_embed
        for i in range(self.N):
            y,_=self.attention(text,text,text)
            text=self.norm1(text+y)
            y=self.ff(text)
            text=self.norm2(text+y)
        return text