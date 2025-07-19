import spacy
from spacy.cli import download


def main():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    doc = nlp("Apple was founded by Steve Jobs in California.")
    for ent in doc.ents:
        print(ent.text, ent.label_)


if __name__ == "__main__":
    main()
