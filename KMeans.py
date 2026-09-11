from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=300, centers=3, random_state=42)

model = KMeans(n_clusters=3, random_state=42)
labels = model.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(model.cluster_centers_[:,0],
            model.cluster_centers_[:,1],
            marker='*', s=300)
plt.show()


