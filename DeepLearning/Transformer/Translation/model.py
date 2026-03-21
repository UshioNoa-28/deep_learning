import torch
import torch.nn as nn
from DeepLearning.Transformer.Decoder import Decoder
from DeepLearning.Transformer.Encoder import Encoder
class Transformer(nn.Module):
    def __init__(self,embed_dim,en_n,de_n,en_fc_size,de_fc_size,vocab_size,max_seq_len,en_num_headers,de_num_headers,state_num):
        super().__init__()
        self.embed_dim=embed_dim
        self.encoder=Encoder(embed_dim,en_n,en_fc_size,vocab_size,max_seq_len,en_num_headers)
        self.decoder=Decoder(de_n,embed_dim,vocab_size,de_num_headers,de_fc_size,max_seq_len)
        self.L=nn.Linear(embed_dim,state_num)
    def forward(self,origin_text,translated_text):
        if self.training:
            en_out=self.encoder(origin_text)
            return self.L(self.decoder(translated_text,en_out))
        else:
            en_out=self.encoder(origin_text)
            batch_size=origin_text.shape[0]
            input_seq=torch.full((batch_size,1),1) # 1为Begin的Id 推理的时候一般batch_size也是1 不过这里严谨一点
            de_out=None
            generated_ids=[]
            while not de_out==2: # 2为End的Id
                new_word=self.L(self.decoder(input_seq,en_out))
                input_seq=torch.concat([input_seq,new_word],dim=1)
                de_out=new_word.argmax(axis=2,keepdims=True).item()
                generated_ids.append(de_out)
            return generated_ids





