# pytorch_rnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 1. 加载和准备数据
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

# 文本向量化
vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(newsgroups_train.data).toarray()
X_test = vectorizer.transform(newsgroups_test.data).toarray()

# 编码标签
encoder = LabelEncoder()
y_train = encoder.fit_transform(newsgroups_train.target)
y_test = encoder.transform(newsgroups_test.target)

# 转换为PyTorch张量并调整形状 (batch_size, seq_len, input_size)
X_train = torch.FloatTensor(X_train).unsqueeze(1)  # 添加序列长度维度
X_test = torch.FloatTensor(X_test).unsqueeze(1)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# 创建数据加载器
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 2. 定义RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        # 前向传播RNN
        out, _ = self.rnn(x, h0)
        
        # 解码最后一个时间步的隐藏状态
        out = self.fc(out[:, -1, :])
        return out

input_size = X_train.shape[2]  # 1000 (词汇大小)
hidden_size = 128
num_classes = len(categories)
model = RNN(input_size, hidden_size, num_classes)

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
epochs = 10
for epoch in range(epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (batch_idx+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 5. 评估模型
correct = 0
total = 0
with torch.no_grad():
    for data, targets in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')