# knn.py
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 加载数据
wine = load_wine()
X = wine.data
y = wine.target

# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建模型
model = KNeighborsClassifier(
    n_neighbors=5, # 1-50 | Number of neighbors
    # weights='uniform', # 'uniform', 'distance'
    # algorithm='auto', # 'ball_tree', 'kd_tree', 'brute'
    # leaf_size=30, # 30-50 | Number of samples in leaf nodes
    # p=2, # Distance metric: 1=Manhattan, 2=Euclidean (default=2)
    # metric='minkowski', # Distance metric (default='minkowski' with p=2 → Euclidean)
    random_state=42
)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")