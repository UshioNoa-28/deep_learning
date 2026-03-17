import torch
import torch.utils.data as data
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
from collections import Counter
import pickle
from DeepLearning.RNN.LSTM.model import LSTMModule
vocab_size=10000
embedding_dim=128
hidden_size=128
nums_layer=3
state_nums=3


net=LSTMModule(vocab_size, embedding_dim, hidden_size, nums_layer, state_nums)

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
max_seq_len = 120
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
# 实际判断时 不能只通过模型判断 模型只认识数据 你还要把词表保存下来
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
print("词表已保存为 vocab.pkl")
def get_accuracy(y_hat,y):
    return (y_hat.argmax(axis=1)==y).type(torch.float).sum()
for epoch in range(epochs):
    test_loss,train_loss=0,0
    test_accu,total=0,0
    net.train()
    for batch in train_iter:
        inputs = batch['input_ids']
        labels = batch['label']
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
            inputs = batch['input_ids']
            labels = batch['label']
            y_hat = net(inputs)
            loss=loss_fn(y_hat,labels)
            test_loss+=loss.item()
            test_accu+=get_accuracy(y_hat,labels)
            total+=len(y_hat)
    print(f"epoch {epoch+1} test loss {test_loss/len(test_iter)} train loss {train_loss/len(train_iter)} accuracy {test_accu/total}")
torch.save(net.state_dict(),"LSTM")

# 正确率薄纱
# epoch 7 test loss 0.4653041107313974 train loss 0.23438666760921478 accuracy 0.8460207581520081
# epoch 8 test loss 0.4722201909337725 train loss 0.14667567610740662 accuracy 0.8549596071243286
# epoch 9 test loss 0.49948071794850485 train loss 0.10151837766170502 accuracy 0.8719723224639893
# epoch 10 test loss 0.5471477934292385 train loss 0.06941282749176025 accuracy 0.8592848777770996
# 实际上并不是模型性能的差异 本身这两个原理差不多性能差不读多的 仅仅是LSTM易于训练
# 实战下来体验可能会差点 17000的数据只有1500个 negative 而且屏蔽脏词 导致很多时候都不会把你归到negative上面
