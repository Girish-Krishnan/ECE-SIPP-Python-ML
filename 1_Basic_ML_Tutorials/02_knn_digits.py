from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main():
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, preds))

    # Confusion matrix visualization
    ConfusionMatrixDisplay.from_predictions(y_test, preds)
    plt.title("k-NN Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
