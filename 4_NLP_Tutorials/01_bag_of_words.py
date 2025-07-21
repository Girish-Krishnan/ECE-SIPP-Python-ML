from sklearn.feature_extraction.text import CountVectorizer


def main():
    docs = [
        "I love coding in Python",
        "Python can be used for NLP"
    ]
    vectorizer = CountVectorizer()
    bag = vectorizer.fit_transform(docs)
    print("Vocabulary:", vectorizer.vocabulary_)
    print("Bag-of-words matrix:\n", bag.toarray())


if __name__ == "__main__":
    main()
