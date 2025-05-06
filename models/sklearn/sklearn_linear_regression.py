# sklearn_linear_regression.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets

# 1. 加载数据集
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 创建模型
model = LinearRegression(
    # fit_intercept=True,
    # copy_X=True,
    # n_jobs=-1
)

# L2
# from sklearn.linear_model import Ridge
# model = Ridge(alpha=1.0)  # alpha是正则化强度

# L1
# from sklearn.linear_model import Lasso 
# model = Lasso(alpha=0.1)  # alpha控制正则化强度

# from sklearn.linear_model import ElasticNet
# model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio控制L1/L2比例

# 4. 训练模型
model.fit(X_train, y_train)

# 5. 预测
y_pred = model.predict(X_test)

# 6. 评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")