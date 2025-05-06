# svm.py
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# For linear kernel
# SVC(kernel='linear', C=0.1)

# For non-linear kernel
# SVC(kernel='rbf', C=1.0, gamma='scale')

# For polynomial kernel (degree >= 3)
# SVC(kernel='poly', degree=4, coef0=1.0)

# 创建模型
model = SVC(
    C=1.0,                # default=1.0，small -> large margin more misclassification
    kernel='rbf',         # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid' (default='rbf')
    # Low gamma → "far" influence (smoother boundaries)
    # High gamma → "close" influence (more complex boundaries)
    # gamma='scale' → 1/(n_features * X.var())
    # gamma='auto' → 1/(n_features)
    gamma='scale',
    degree=3,             # Polynomial degree (only for kernel='poly')
    coef0=0.0,            # Independent term in kernel (poly/sigmoid)
    class_weight=None,    # Handle class imbalance: 'balanced' or dict
    probability=False,    # Enable probability estimates (slows training)
    random_state=42       # Seed for stochastic parts
)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'C': [0.1, 1, 10],
#     'gamma': [0.01, 0.1, 1],
#     'kernel': ['rbf', 'linear']
# }
# grid = GridSearchCV(SVC(), param_grid, cv=5)
# grid.fit(X_train, y_train)