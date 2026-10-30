import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

iris = load_iris(as_frame=True)
df = iris.data
X = StandardScaler().fit_transform(df)

# Elbow method (K-Means) and Dendrogram (Hierarchical)
ks = range(1, 11)
inertia = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_ for k in ks]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ks, inertia, "o-")
ax[0].set(title="Elbow Method", xlabel="k", ylabel="Inertia")
dendrogram(linkage(X, method="ward"), ax=ax[1], truncate_mode="lastp", p=20)
ax[1].set(title="Dendrogram (Ward)", xlabel="Samples", ylabel="Distance")
plt.tight_layout()
plt.show()

# Fit both models with chosen k
k = 3
km_labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
hc_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)

print("Silhouette  K-Means:", round(silhouette_score(X, km_labels), 4))
print("Silhouette  Hierarchical:", round(silhouette_score(X, hc_labels), 4))

# Cluster analysis
df["KMeans"], df["Hierarchical"] = km_labels, hc_labels
print("\nK-Means cluster means:\n", df.groupby("KMeans").mean().round(2))
print("\nHierarchical cluster means:\n", df.groupby("Hierarchical").mean().round(2))
print("\nCluster sizes:\n", df[["KMeans", "Hierarchical"]].apply(pd.Series.value_counts))

# Visualize clusters in 2D (PCA)
pts = PCA(n_components=2).fit_transform(X)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, labels, title in zip(ax, [km_labels, hc_labels], ["K-Means", "Hierarchical"]):
    a.scatter(pts[:, 0], pts[:, 1], c=labels, cmap="viridis")
    a.set(title=title, xlabel="PC1", ylabel="PC2")
plt.tight_layout()
plt.show()