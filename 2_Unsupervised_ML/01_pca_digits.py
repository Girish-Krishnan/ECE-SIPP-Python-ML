from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def pca_scratch(X: np.ndarray, n_components: int = 2):
    """Principal component analysis using eigendecomposition."""
    X_c = X - X.mean(axis=0)
    cov = np.cov(X_c, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    components = eigvecs[:, :n_components]
    explained = eigvals[:n_components] / eigvals.sum()
    reduced = X_c @ components
    return reduced, explained


def main():
    X, y = load_digits(return_X_y=True)
    reduced, var_ratio = pca_scratch(X, n_components=2)
    print("From-scratch explained variance ratio:", var_ratio)
    print("Reduced shape:", reduced.shape)

    pca = PCA(n_components=2)
    sk_reduced = pca.fit_transform(X)
    print("scikit-learn explained variance ratio:", pca.explained_variance_ratio_)
    print("scikit-learn transformed shape:", sk_reduced.shape)

    # Scatter plot of the first two principal components
    plt.scatter(sk_reduced[:, 0], sk_reduced[:, 1], c=y, cmap="tab10", s=15)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Digits")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
