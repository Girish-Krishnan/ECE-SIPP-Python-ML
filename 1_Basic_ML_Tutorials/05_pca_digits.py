from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def main():
    X, y = load_digits(return_X_y=True)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Transformed shape:", reduced.shape)

    # Scatter plot of the first two principal components
    plt.scatter(reduced[:, 0], reduced[:, 1], c=y, cmap="tab10", s=15)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Digits")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
