import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from collections import Counter

# 加载数据
transform = transforms.ToTensor()
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# 取一张图像
image, label = dataset[0]

# Tensor 转换为 numpy (注意 image 是 [1, 28, 28])
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()

import matplotlib.pyplot as plt

# # 展示前 16 张图像
# fig, axes = plt.subplots(4, 4, figsize=(6, 6))
# for i in range(16):
#     image, label = dataset[i]
#     ax = axes[i // 4, i % 4]
#     ax.imshow(image.squeeze(), cmap='gray')
#     ax.set_title(label)
#     ax.axis('off')

# plt.tight_layout()
# plt.show()

# 提取所有标签
labels = [label for _, label in dataset]

# 统计标签频率
counter = Counter(labels)
classes = list(range(10))  # 0 到 9
frequencies = [counter[i] for i in classes]

# 可视化
plt.bar(classes, frequencies)
plt.xlabel("Digit Label")
plt.ylabel("Frequency")
plt.title("MNIST Label Distribution")
plt.xticks(classes)
plt.show()

