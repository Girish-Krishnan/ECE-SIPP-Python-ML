from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def main():
    X, y = load_iris(return_X_y=True)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    print("Cluster counts:", np.bincount(clusters))
    print("Cluster centers:\n", kmeans.cluster_centers_)

    # Scatter plot of the first two features with cluster assignments
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap="viridis", s=30)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c="red", marker="x", s=100, linewidths=2, label="Centers")
    plt.title("k-means Clustering")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
