import nltk


def main():
    nltk.download('punkt', quiet=True)
    text = "Natural language processing with Python is fun!"
    tokens = nltk.word_tokenize(text)
    print(tokens)


if __name__ == "__main__":
    main()
