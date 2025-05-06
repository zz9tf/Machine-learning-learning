# sklearn_logistic_regression.py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 加载数据集
data = load_breast_cancer()
X = data.data
y = data.target

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 创建模型
model = LogisticRegression(
    # penalty='l1', # Options: 'l1', 'l2', 'elasticnet', 'none'
    # C=0.1, # Inverse of regularization strength; must be a positive float.
    # solver='saga', # Options: l1: 'liblinear'(elasticnet), 'saga', l2: 'lbfgs' | 'newton-cg', 'newton-cholesky', 'sag'
    # tol=1e-4, # Tolerance for stopping criteria
    # max_iter=5000,
    # class_weight='balanced' # Handle class imbalance: 'balanced' or custom weights, default=None
)

# 4. 训练模型
model.fit(X_train, y_train)

# 5. 预测
y_pred = model.predict(X_test)

# 6. 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))