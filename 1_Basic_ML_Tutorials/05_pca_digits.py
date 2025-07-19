from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


def main():
    X, _ = load_digits(return_X_y=True)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Transformed shape:", reduced.shape)


if __name__ == "__main__":
    main()
