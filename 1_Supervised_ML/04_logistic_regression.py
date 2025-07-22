import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression_gd(X, y, lr=0.1, epochs=200):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        preds = sigmoid(z)
        grad_w = X.T @ (preds - y) / len(y)
        grad_b = np.mean(preds - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def main():
    X, y = load_iris(return_X_y=True)
    mask = y < 2
    X = X[mask, :2]
    y = y[mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    w, b = logistic_regression_gd(X_train, y_train)
    preds = (sigmoid(X_test @ w + b) >= 0.5).astype(int)
    print("From-scratch accuracy:", accuracy_score(y_test, preds))

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    sk_preds = clf.predict(X_test)
    print("scikit-learn accuracy:", accuracy_score(y_test, sk_preds))


if __name__ == "__main__":
    main()
