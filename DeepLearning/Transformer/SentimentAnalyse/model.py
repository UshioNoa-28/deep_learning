import torch.nn as nn
from DeepLearning.Transformer.Decoder import Decoder
from DeepLearning.Transformer.Encoder import Encoder
class Transformer(nn.Module):
    def __init__(self,embed_dim,en_n,de_n,en_fc_size,de_fc_size,vocab_size,max_seq_len,en_num_headers,de_num_headers,state_num):
        super().__init__()
        self.encoder=Encoder(embed_dim,en_n,en_fc_size,vocab_size,max_seq_len,en_num_headers)
        self.decoder=Decoder(de_n,embed_dim,vocab_size,de_num_headers,de_fc_size,max_seq_len)
        self.L=nn.Linear(embed_dim,state_num)
    def forward(self,text):
        en_out=self.encoder(text)
        text=self.decoder(text,en_out)
        text=text.mean(dim=1)
        return self.L(text)
