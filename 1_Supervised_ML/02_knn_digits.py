import numpy as np
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def knn_predict(X_train, y_train, X_test, k=3):
    preds = []
    for x in X_test:
        dists = np.linalg.norm(X_train - x, axis=1)
        idx = np.argsort(dists)[:k]
        preds.append(np.bincount(y_train[idx]).argmax())
    return np.array(preds)


def main():
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    preds = knn_predict(X_train, y_train, X_test, k=3)
    print("From-scratch accuracy:", accuracy_score(y_test, preds))

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    sk_preds = knn.predict(X_test)
    print("scikit-learn accuracy:", accuracy_score(y_test, sk_preds))


if __name__ == "__main__":
    main()
