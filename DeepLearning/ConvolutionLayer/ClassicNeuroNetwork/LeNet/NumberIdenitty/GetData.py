import torch
from torchvision import datasets, transforms
import torch.utils.data as data

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def get_data(batch_size):
    return data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True),data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)