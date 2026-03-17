import torch
import torch.nn.functional as F
import pickle
import os
from model import LSTMModule  # 直接引用你的模型类

# region 配置参数 (必须与训练时完全一致)
MODEL_PATH = "LSTM"
VOCAB_PATH = "vocab.pkl"  # 建议你把训练时的 vocab 存成这个文件
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 3
STATE_NUMS = 3
MAX_LEN = 120
# endregion

# region 1. 加载必要的组件
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(VOCAB_PATH):
    print(f"错误：找不到词表文件 {VOCAB_PATH}。请先在训练脚本中保存词表。")
    exit()

with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)

# 初始化模型结构
net = LSTMModule(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=HIDDEN_SIZE,
    nums_layer=NUM_LAYERS,
    state_nums=STATE_NUMS
)

# 加载权重
net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
net.to(device)
net.eval()  # 开启预测模式（关闭 Dropout 和 BatchNorm）

# 标签映射
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}


# endregion

# region 2. 预测函数
def get_sentiment(text):
    # 简单的预处理：分词 -> ID化 -> Padding
    tokens = [vocab.get(w.lower(), vocab.get('<UNK>', 1)) for w in str(text).split()]

    if len(tokens) < MAX_LEN:
        tokens += [vocab.get('<PAD>', 0)] * (MAX_LEN - len(tokens))
    else:
        tokens = tokens[:MAX_LEN]

    # 增加 batch 维度并传给设备
    input_tensor = torch.tensor([tokens]).to(device)

    with torch.no_grad():
        logits = net(input_tensor)
        probs = F.softmax(logits, dim=1)  # 转化为概率

    return probs[0]


# endregion

# region 3. 控制台交互
print("=" * 30)
print("  LSTM 评价情感分析系统  ")
print("  (输入 'q' 退出)  ")
print("=" * 30)

while True:
    user_input = input("\n请输入用户评价: ").strip()

    if user_input.lower() == 'q':
        break
    if not user_input:
        continue

    probs = get_sentiment(user_input)

    # 打印每个类别的概率
    print("-" * 20)
    for i, p in enumerate(probs):
        label = id2label[i]
        print(f"{label.ljust(10)}: {p.item() * 100:>6.2f}%")

    top_idx = torch.argmax(probs).item()
    print("-" * 20)
    print(f"最终判定: 【{id2label[top_idx]}】")

print("\n程序已退出。")
# endregion