from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, preds))


if __name__ == "__main__":
    main()
