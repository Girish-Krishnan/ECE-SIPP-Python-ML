import spacy
from spacy.cli import download


def main():
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        download("en_core_web_md")
        nlp = spacy.load("en_core_web_md")

    tokens = nlp("dog cat banana")
    for token1 in tokens:
        for token2 in tokens:
            print(f"Similarity({token1.text}, {token2.text}) = {token1.similarity(token2):.3f}")


if __name__ == "__main__":
    main()
