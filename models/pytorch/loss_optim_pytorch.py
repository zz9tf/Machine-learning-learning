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

# 1. Multi-class classification (e.g., MNIST digit recognition)
# It has a built-in Softmax function.
ce_loss = nn.CrossEntropyLoss()
loss1 = ce_loss(predictions, targets)

# 2. Binary classification (e.g., binary MNIST classification)
# Suitable for sigmoid outputs.
pred_probs = torch.sigmoid(torch.randn(10, 1, requires_grad=True))
binary_targets = torch.randint(0, 2, (10, 1)).float()
bce_loss = nn.BCELoss()
loss2 = bce_loss(pred_probs, binary_targets)

# 3. Binary classification + Logits: BCEWithLogitsLoss (More stable)
logits = torch.randn(10, 1, requires_grad=True)
bce_logits_loss = nn.BCEWithLogitsLoss()
loss3 = bce_logits_loss(logits, binary_targets)

# 4. Regression: MSELoss (Mean Squared Error)
pred_vals = torch.randn(10, 1, requires_grad=True)
true_vals = torch.randn(10, 1)
mse_loss = nn.MSELoss()
loss4 = mse_loss(pred_vals, true_vals)

# 5. Regression: L1Loss (Absolute Error)
l1_loss = nn.L1Loss()
loss5 = l1_loss(pred_vals, true_vals)


# ====================
# Optimizers
# ====================

model = nn.Linear(10, 2)  # 假设的模型

# 1. SGD (Stochastic Gradient Descent) 
# Simple convex problems
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01)

# 2. SGD + Momentum (Convex problems) 
# Accelerates convergence in ravines
optimizer_sgdm = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 3. Adam (Non-convex problems) General-purpose
# recommended for most cases
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)

# 4. RMSprop (Non-convex problems) 
# Good for recurrent networks
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.001)

# Example usage
optimizer_adam.zero_grad()
loss1.backward()
optimizer_adam.step()


# ====================
# Activation Functions
# ====================

# Create input tensor for activation examples
x = torch.randn(5)  # values range from negative to positive
x_pos = torch.rand(5)  # only positive values for probability functions

# 1. ReLU (Rectified Linear Unit)
# Default for most hidden layers; solves vanishing gradient problem
relu = nn.ReLU()
relu_output = relu(x)
print(f"ReLU: {relu_output}")  # negative values become 0

# 2. Leaky ReLU
# Prevents "dying ReLU" by allowing small negative values
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
leaky_output = leaky_relu(x)
print(f"Leaky ReLU: {leaky_output}")  # negative values scaled by 0.01

# 3. Sigmoid
# Maps any value to (0,1) range; used for binary classification outputs
sigmoid = nn.Sigmoid()
sigmoid_output = sigmoid(x)
print(f"Sigmoid: {sigmoid_output}")  # values between 0 and 1

# 4. Tanh
# Maps any value to (-1,1) range; often used in RNNs
tanh = nn.Tanh()
tanh_output = tanh(x)
print(f"Tanh: {tanh_output}")  # values between -1 and 1

# 5. Softmax
# Converts values to probability distribution; used for multi-class outputs
softmax = nn.Softmax(dim=1) # [B, C] -> [B, C] softmax on C
softmax_output = softmax(x)
print(f"Softmax: {softmax_output}")  # values sum to 1.0

# 6. GELU (Gaussian Error Linear Unit)
# Used in transformers (BERT, GPT); smooth approximation of ReLU
gelu = nn.GELU()
gelu_output = gelu(x)
print(f"GELU: {gelu_output}")