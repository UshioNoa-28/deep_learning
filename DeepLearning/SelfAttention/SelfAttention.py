import torch
import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
from collections import Counter
from Model import SelfAttentionModule
# 实际上整个实验都是直接照办过来的 来看看二者差距有多大
vocab_size=10000
embedding_dim=128
nums_header=4
hidden_layer1=128
hidden_layer2=256
state_nums=3
max_seq_len = 120
net=SelfAttentionModule(embedding_dim,nums_header,hidden_layer1,state_nums,vocab_size,max_seq_len)
net=net.to("cuda")
# region 下面为数据处理部分
df = pd.read_csv('cleaned_reviews.csv')
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df['label'] = df['sentiments'].map(label_map)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


def build_vocab(sentences, max_vocab_size=5000):
    all_words = []
    for s in sentences:
        all_words.extend(str(s).split())
    counts = Counter(all_words)
    vocab = {word: i + 2 for i, (word, _) in enumerate(counts.most_common(max_vocab_size-2))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab
vocab = build_vocab(train_df['cleaned_review'],max_vocab_size=vocab_size)
class ReviewDataset(data.Dataset):
    def __init__(self, df, vocab, max_len=50):
        self.labels = df['label'].values
        self.vocab = vocab
        self.max_len = max_len
        self.sequences = []
        for text in df['cleaned_review']:
            words = str(text).split()
            seq = [vocab.get(w, vocab['<UNK>']) for w in words]
            if len(seq) < max_len:
                seq += [vocab['<PAD>']] * (max_len - len(seq))
            else:
                seq = seq[:max_len]
            self.sequences.append(seq)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.sequences[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = ReviewDataset(train_df, vocab, max_len=max_seq_len)
test_dataset = ReviewDataset(test_df, vocab, max_len=max_seq_len)
# end region
batch_size=256
train_iter=data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_iter=data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
epochs=9
lr=0.002
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(lr=lr,params=net.parameters())

def get_accuracy(y_hat,y):
    return (y_hat.argmax(axis=1)==y).type(torch.float).sum()
for epoch in range(epochs):
    test_loss,train_loss=0,0
    test_accu,total=0,0
    net.train()
    for batch in train_iter:
        inputs = batch['input_ids'].to("cuda")
        labels = batch['label'].to("cuda")
        y_hat=net(inputs)
        optimizer.zero_grad()
        loss=loss_fn(y_hat,labels)
        train_loss+=loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),max_norm=6)
        optimizer.step()
    net.eval()
    with torch.no_grad():
        for batch in test_iter:
            inputs = batch['input_ids'].to("cuda")
            labels = batch['label'].to("cuda")
            y_hat = net(inputs)
            loss=loss_fn(y_hat,labels)
            test_loss+=loss.item()
            test_accu+=get_accuracy(y_hat,labels)
            total+=len(y_hat)
    print(f"epoch {epoch+1} test loss {test_loss/len(test_iter)} train loss {train_loss/len(train_iter)} accuracy {test_accu/total}")
torch.save(net.state_dict(),"SelfAttention")