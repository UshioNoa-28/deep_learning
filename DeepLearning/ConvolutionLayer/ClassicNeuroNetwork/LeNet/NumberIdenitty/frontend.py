import torch
import torch.nn as nn
import gradio as gr
import numpy as np
from PIL import Image, ImageOps


# 1. 模型结构定义 (保持和你训练时完全一致)
class Reshape(nn.Module):
    def forward(self, X):
        return X.reshape((-1, 1, 32, 32))


net = nn.Sequential(
    Reshape(),
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(in_features=16 * 5 * 5, out_features=120),
    nn.ReLU(),
    nn.Linear(in_features=120, out_features=84),
    nn.ReLU(),
    nn.Linear(in_features=84, out_features=10)
)

# 2. 加载权重
try:
    net.load_state_dict(torch.load("LeNet", map_location=torch.device('cpu')))
    net.eval()
    print("模型权重加载成功！")
except Exception as e:
    print(f"权重加载失败: {e}")


# 3. 预测函数
def predict(image_data):
    if image_data is None: return None

    # 从字典中提取图片
    img = image_data["composite"]

    # 预处理：转灰度 -> 反转颜色(适配MNIST) -> 缩放 -> 归一化
    img = img.convert('L')
    img = ImageOps.invert(img)
    img = img.resize((32, 32), Image.Resampling.LANCZOS)

    tensor = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        logits = net(tensor)
        probs = torch.nn.functional.softmax(logits[0], dim=0)

    return {str(i): float(probs[i]) for i in range(10)}


# 4. 界面
demo = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(label="手写数字", type="pil"),  # 指定 type="pil" 更方便
    outputs=gr.Label(num_top_classes=3),
    live=True
)

demo.launch()