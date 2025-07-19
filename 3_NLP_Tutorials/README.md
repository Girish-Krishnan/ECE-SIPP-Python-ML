# Natural Language Processing (NLP) Tutorials

This folder contains short examples showing how to process and analyze text. Each script can be executed with `python` followed by the file name.

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

Feel free to modify these examples or use them as starting points for your own NLP experiments. Additional packages like `nltk`, `transformers` and `spacy` can be installed with pip if they are not already available.
