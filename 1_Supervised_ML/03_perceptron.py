import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


def perceptron_gd(X, y, lr=0.1, epochs=100):
    y_signed = np.where(y == 0, -1, 1)
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        margins = y_signed * (X @ w + b)
        mask = margins < 0
        if not mask.any():
            break
        grad_w = -(y_signed[mask, None] * X[mask]).mean(axis=0)
        grad_b = -(y_signed[mask]).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def main():
    X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                               n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    w, b = perceptron_gd(X_train, y_train)
    preds = (X_test @ w + b >= 0).astype(int)
    print("From-scratch accuracy:", accuracy_score(y_test, preds))

    model = Perceptron(max_iter=1000, eta0=0.1, tol=1e-3)
    model.fit(X_train, y_train)
    sk_preds = model.predict(X_test)
    print("scikit-learn accuracy:", accuracy_score(y_test, sk_preds))


if __name__ == "__main__":
    main()
