import torch
from torch.utils.data import Dataset, DataLoader
import re

from DeepLearning.Transformer.Translation.model import Transformer


class Vocab:
    def __init__(self, sentences, is_chinese=False):
        # 0:PAD, 1:BEG, 2:END, 3:UNK
        self.itos = {0: "<PAD>", 1: "<BEG>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

        for sent in sentences:
            tokens = list(sent) if is_chinese else sent.lower().split()
            for token in tokens:
                if token not in self.stoi:
                    idx = len(self.stoi)
                    self.stoi[token] = idx
                    self.itos[idx] = token

    def __len__(self):
        return len(self.stoi)

    def encode(self, text, is_chinese=False, max_len=20):
        tokens = list(text) if is_chinese else text.lower().split()
        ids = [self.stoi.get(t, 3) for t in tokens]
        return ids


class cmnDataset(Dataset):
    def __init__(self, pairs, en_vocab, zh_vocab, max_len=120):
        self.pairs = pairs
        self.en_vocab = en_vocab
        self.zh_vocab = zh_vocab
        self.max_len = max_len

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        en_text, zh_text = self.pairs[idx]

        # Encoder 输入: [I, love, cats] -> [10, 11, 12, 0, 0...]
        en_ids = self.en_vocab.encode(en_text, is_chinese=False)
        en_ids = (en_ids + [0] * self.max_len)[:self.max_len]

        # Decoder 完整序列: [<BEG>, 我, 爱, 猫, <END>]
        zh_ids = self.zh_vocab.encode(zh_text, is_chinese=True)
        zh_full = ([1] + zh_ids + [2] + [0] * self.max_len)[:self.max_len + 1]

        # 训练三剑客：输入、预测输入、标签
        return torch.tensor(en_ids).to("cuda"), torch.tensor(zh_full[:-1]).to("cuda"), torch.tensor(zh_full[1:]).to("cuda")


def get_translation_dataloader(batch_size=32,max_seq_len=120):
    # 1. 读取并清洗数据 (cmn.txt 格式: EN \t ZH \t Attribution)
    pairs = []
    with open('cmn.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split('\t')
            if len(parts) >= 2:
                en = re.sub(r'[.,!?]', '', parts[0])  # 简单去标点
                zh = parts[1]
                pairs.append((en, zh))

    # 2. 构建词表
    en_vocab = Vocab([p[0] for p in pairs], is_chinese=False)
    zh_vocab = Vocab([p[1] for p in pairs], is_chinese=True)

    # 3. 创建 DataLoader
    dataset = cmnDataset(pairs, en_vocab, zh_vocab,max_len=max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader, en_vocab, zh_vocab


embed_dim=128
max_seq_len=120
loader, en_vocab, zh_vocab = get_translation_dataloader(batch_size=128,)


# 注意：vocab_size 要传给你的 Transformer
model = Transformer(embed_dim=embed_dim,en_n=6,de_n=6,en_fc_size=64,de_fc_size=64,vocab_size=max(len(en_vocab),len(zh_vocab)),
                    max_seq_len=max_seq_len,en_num_headers=8,de_num_headers=8,state_num=len(zh_vocab))
model=model.to("cuda")
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)  # 别忘了忽略 PAD
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    for src, dec_in, label in loader:
        output = model(src, dec_in)  # 输出: [Batch, seq_len, vocab_size]
        # 展平算 Loss
        loss = criterion(output.view(-1, output.size(-1)), label.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item()}")

# 训练完保存
torch.save({
    'model_state': model.state_dict(),
    'en_stoi': en_vocab.stoi,
    'zh_itos': zh_vocab.itos
}, 'transformer_cmn.pth')
# Epoch 10 Loss: 2.876842737197876
# Epoch 11 Loss: 2.684591770172119
# Epoch 12 Loss: 2.8069543838500977
# Epoch 13 Loss: 2.4934499263763428
# Epoch 14 Loss: 2.8247389793395996
# Epoch 15 Loss: 2.2522013187408447
# Epoch 16 Loss: 2.2999062538146973
# Epoch 17 Loss: 2.1738362312316895
# Epoch 18 Loss: 1.8970251083374023
# Epoch 19 Loss: 2.3266971111297607

# Input English:  i miss you
# Output Chinese: 我想念你的。
# Input English:  i love you
# Output Chinese: 我爱你的意思。
# Input English:  how are you
# Output Chinese: 你是谁的？
# Input English:  who are you
# Output Chinese: 你是誰的？
# Input English:  why you love him
# Output Chinese: 你認為他的問題嗎？
# Input English:  do you love me
# Output Chinese: 你是我愛的吗？
# Input English:  i have a cat
# Output Chinese: 我有一個小心。
# Input English:  be careful
# Output Chinese: 我们要擔心。
# Input English:  you are right
# Output Chinese: 你是对的。
# 翻译效果我只能说 有点人样