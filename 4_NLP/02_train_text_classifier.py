from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report


def main():
    categories = ['rec.sport.baseball', 'sci.space']
    train = fetch_20newsgroups(subset='train', categories=categories,
                              remove=('headers', 'footers', 'quotes'))
    test = fetch_20newsgroups(subset='test', categories=categories,
                             remove=('headers', 'footers', 'quotes'))

    clf = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
    clf.fit(train.data, train.target)
    preds = clf.predict(test.data)
    print(classification_report(test.target, preds, target_names=test.target_names))


if __name__ == "__main__":
    main()
