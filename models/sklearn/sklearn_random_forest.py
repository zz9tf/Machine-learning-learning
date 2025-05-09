# random_forest.py
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 加载数据
data = load_breast_cancer()
X = data.data
y = data.target

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建模型
model = RandomForestClassifier(
        n_estimators=200, # 100–500
        # max_depth=15, # 5–30
        # max_features=0.8, # 'sqrt', 0.6–0.8
        # min_samples_leaf=5, # 2–20 Prevent overfitting by restricting node splits
        # class_weight='balanced',
        # n_jobs=-1,
        random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))