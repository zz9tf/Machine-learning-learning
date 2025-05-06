# 常用的 Loss Functions 和 Optimizers in PyTorch

import torch
import torch.nn as nn
import torch.optim as optim

# 假设输出和标签（仅为演示用途）
predictions = torch.randn(10, 5, requires_grad=True)  # [batch_size, num_classes]
targets = torch.randint(0, 5, (10,))  # [batch_size]

# ====================
# Loss Functions
# ====================

# 1. 多分类：CrossEntropyLoss（内部自带Softmax）
ce_loss = nn.CrossEntropyLoss()
loss1 = ce_loss(predictions, targets)

# 2. 二分类：BCELoss (适用于 sigmoid 输出)
pred_probs = torch.sigmoid(torch.randn(10, 1, requires_grad=True))
binary_targets = torch.randint(0, 2, (10, 1)).float()
bce_loss = nn.BCELoss()
loss2 = bce_loss(pred_probs, binary_targets)

# 3. 二分类 + Logits：BCEWithLogitsLoss（更稳定）
logits = torch.randn(10, 1, requires_grad=True)
bce_logits_loss = nn.BCEWithLogitsLoss()
loss3 = bce_logits_loss(logits, binary_targets)

# 4. 回归：MSELoss（均方误差）
pred_vals = torch.randn(10, 1, requires_grad=True)
true_vals = torch.randn(10, 1)
mse_loss = nn.MSELoss()
loss4 = mse_loss(pred_vals, true_vals)

# 5. 回归：L1Loss（绝对值误差）
l1_loss = nn.L1Loss()
loss5 = l1_loss(pred_vals, true_vals)


# ====================
# Optimizers
# ====================

model = nn.Linear(10, 2)  # 假设的模型

# 1. SGD
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01)

# 2. SGD + Momentum
optimizer_sgdm = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 3. Adam
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)

# 4. RMSprop
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.001)

# 使用示例
optimizer_adam.zero_grad()
loss1.backward()
optimizer_adam.step()
