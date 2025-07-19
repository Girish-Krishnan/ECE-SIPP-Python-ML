from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np


def main():
    X, y = load_iris(return_X_y=True)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    print("Cluster counts:", np.bincount(clusters))
    print("Cluster centers:\n", kmeans.cluster_centers_)


if __name__ == "__main__":
    main()
