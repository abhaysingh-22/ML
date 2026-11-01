import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)  # 13 chemical features, 3 wine classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

# PCA keeping 95% of variance (fit on train only)
pca = PCA(n_components=0.95).fit(X_train)
Xp_train, Xp_test = pca.transform(X_train), pca.transform(X_test)
print(f"Features before PCA: {X_train.shape[1]}")
print(f"Features after PCA : {Xp_train.shape[1]}")

# Classification before vs after PCA
for name, (a, b) in {"Original": (X_train, X_test), "PCA": (Xp_train, Xp_test)}.items():
    t = time.perf_counter()
    model = LogisticRegression(max_iter=2000).fit(a, y_train)
    acc = accuracy_score(y_test, model.predict(b))
    print(f"{name:<9} accuracy: {acc:.4f} | time: {(time.perf_counter() - t) * 1000:.1f} ms")

# Visualization
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(np.cumsum(PCA().fit(X_train).explained_variance_ratio_), "o-")
ax[0].axhline(0.95, color="r", linestyle="--")
ax[0].set(title="Cumulative Explained Variance", xlabel="Components", ylabel="Variance")

p2 = PCA(n_components=2).fit_transform(scaler.transform(X))
sc = ax[1].scatter(p2[:, 0], p2[:, 1], c=y, cmap="viridis", s=15)
ax[1].set(title="Wine in 2D (PCA)", xlabel="PC1", ylabel="PC2")
fig.colorbar(sc, ax=ax[1], label="Wine class")

plt.tight_layout()
plt.show()