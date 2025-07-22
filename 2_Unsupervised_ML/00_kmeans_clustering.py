from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def kmeans_scratch(X: np.ndarray, k: int, max_iter: int = 100, random_state: int = 42):
    """Simple k-means implementation using NumPy."""
    rng = np.random.default_rng(random_state)
    centroids = X[rng.choice(len(X), size=k, replace=False)]
    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, None] - centroids[None], axis=2)
        labels = distances.argmin(axis=1)
        new_centroids = np.array([
            X[labels == i].mean(axis=0) for i in range(k)
        ])
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return labels, centroids


def main():
    X, _ = load_iris(return_X_y=True)
    labels, centers = kmeans_scratch(X, k=3)
    print("From-scratch cluster counts:", np.bincount(labels))
    print("Centroids from scratch:\n", centers)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    sk_labels = kmeans.fit_predict(X)
    print("scikit-learn cluster counts:", np.bincount(sk_labels))
    print("scikit-learn centers:\n", kmeans.cluster_centers_)

    # Scatter plot of the first two features with cluster assignments
    plt.scatter(X[:, 0], X[:, 1], c=sk_labels, cmap="viridis", s=30)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c="red", marker="x", s=100, linewidths=2, label="Centers")
    plt.title("k-means Clustering")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
