from transformers import pipeline


def main():
    classifier = pipeline("sentiment-analysis")
    text = "I love using transformers for NLP!"
    result = classifier(text)[0]
    print(f"Label: {result['label']}, score: {result['score']:.3f}")


if __name__ == "__main__":
    main()
