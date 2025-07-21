# Natural Language Processing (NLP) Tutorials

There are two main ways to run the examples in this directory:

## Option 1: Jupyter Notebook

Click on the badge below to open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Girish-Krishnan/ECE-SIPP-Python-ML/blob/main/3_NLP_Tutorials/nlp_tutorials.ipynb)

The notebook walks through each task step by step with explanations.

## Option 2: Python Scripts

You can also execute the scripts individually with `python`. Make sure the required packages are installed:

```bash
pip install nltk scikit-learn transformers spacy
```

### 0. Tokenize text with NLTK
`python 00_tokenize_text.py`

Downloads the NLTK tokenizer data and splits a sentence into tokens.

### 1. Bag-of-words representation
`python 01_bag_of_words.py`

Uses scikit-learn to build a simple bag-of-words matrix from two sentences.

### 2. Train a text classifier
`python 02_train_text_classifier.py`

Loads two categories from the 20 Newsgroups dataset and fits a logistic regression classifier. A classification report on the test split is printed.

### 3. Sentiment analysis with transformers
`python 03_sentiment_transformers.py`

Runs a Hugging Face `pipeline` to classify a short sentence as positive or negative.

### 4. Named entity recognition with spaCy
`python 04_named_entity_recognition.py`

Loads the small English spaCy model and prints entities found in a sample sentence.

### 5. Word embeddings and similarity
`python 05_word_embeddings.py`

Loads medium-size spaCy vectors and prints pairwise similarities between a few sample words.

Feel free to modify these examples or use them as starting points for your own NLP experiments.
