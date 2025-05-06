# kmeans.py
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 生成数据
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 应用KMeans
model = KMeans(
    n_clusters=4, # 2-20 | Domain knowledge/elbow method
    # max_iter=300, # 10-500 | Convergence, For large/complex datasets
    # n_init=10, # 1-20 | Number of initializations, Ensure stable centroids
    random_state=42
)
model.fit(X)
y_kmeans = model.predict(X)

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = model.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title('K-means Clustering')
plt.show()