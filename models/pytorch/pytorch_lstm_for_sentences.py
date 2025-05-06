import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from collections import defaultdict
import random

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模拟文本数据和标签
texts = [
    "I love AI",
    "This is powerful",
    "Hello world",
    "Deep learning is amazing",
    "AI is the future",
    "PyTorch is great",
    "Transformers are useful"
]
labels = [1, 0, 1, 1, 0, 1, 0]  # 示例：二分类

# 构建词表
word2idx = defaultdict(lambda: len(word2idx))
word2idx["<PAD>"] = 0
word2idx["<UNK>"] = 1

tokenized = []
for text in texts:
    tokens = text.lower().split()
    token_ids = [word2idx[token] for token in tokens]
    tokenized.append(torch.tensor(token_ids))

lengths = torch.tensor([len(seq) for seq in tokenized])
padded_seqs = pad_sequence(tokenized, batch_first=True, padding_value=0)
labels_tensor = torch.tensor(labels)

# 自定义 Dataset
class TextDataset(Dataset):
    def __init__(self, sequences, labels, lengths):
        self.sequences = sequences
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.lengths[idx]

# 划分 train/test
data = list(zip(padded_seqs, labels_tensor, lengths))
random.shuffle(data)
split = int(0.8 * len(data))
train_data = data[:split]
test_data = data[split:]

train_dataset = TextDataset(*zip(*train_data))
test_dataset = TextDataset(*zip(*test_data))

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

# 模型定义
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)  # [B, T] -> [B, T, E]
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        out = self.fc(h_n[-1])  # [B, hidden_size] -> [B, num_classes]
        return out

# 超参数
vocab_size = len(word2idx)
embed_dim = 64
hidden_size = 128
num_layers = 2
num_classes = 2
learning_rate = 0.001
epochs = 10

model = TextLSTM(vocab_size, embed_dim, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练
model.train()
for epoch in range(epochs):
    total_loss = 0
    for x, y, lengths in train_loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        outputs = model(x, lengths)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# 测试
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y, lengths in test_loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        outputs = model(x, lengths)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
