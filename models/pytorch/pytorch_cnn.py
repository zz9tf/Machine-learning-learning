# pytorch_cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. 加载和准备数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# 2. 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # 输入: [B, 1, 28, 28] (B为批次大小)
        x = self.conv1(x) # conv1: [B, 32, 26, 26] (28x28 → 26x26，3x3卷积核，步长1)
        x = nn.ReLU()(x)
        x = self.conv2(x) # conv2: [B, 64, 24, 24] (26x26 → 24x24，3x3卷积核，步长1)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x) # MaxPool2d: [B, 64, 12, 12] (24x24 → 12x12，2x2池化)
        x = self.dropout1(x) # dropout1: [B, 64, 12, 12] (维度不变)
        x = torch.flatten(x, 1) # flatten: [B, 9216] (64×12×12 = 9216)
        x = self.fc1(x) # fc1: [B, 128]
        x = nn.ReLU()(x) 
        x = self.dropout2(x) # dropout2: [B, 128] (维度不变)
        x = self.fc2(x) # fc2: [B, 10]
        return nn.LogSoftmax(dim=1)(x) # LogSoftmax输出: [B, 10]

model = CNN()

# 3. 定义损失函数和优化器
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')

# 5. 评估模型
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')