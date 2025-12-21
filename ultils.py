# model_utils.py
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load NLP tools
lemm = WordNetLemmatizer()

def remove_punctuation_func(text):
    return re.sub(r'[^a-zA-Z0-9 ]', '', text)


def toLower(data):
	if isinstance(data, float):
		return '<UNK>'
	else:
		return data.lower()

def remove_punctuation_func(text):
	return re.sub(r'[^a-zA-Z0-9]', ' ', text)

def preprocess_input(text, tokenizer):
    text = toLower(text)
    text = remove_punctuation_func(text)
    text = " ".join([lemm.lemmatize(w) for w in text.split()])

    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=40, padding='post', truncating='post')
    return pad


def predict_review(text, model, tokenizer):
    x = preprocess_input(text, tokenizer)
    prob = model.predict(x)[0][0]

    if prob >= 0.7:
        label = "Recommended 👍"
    elif prob <= 0.3:
        label = "Not recommended 👎"
    else:
        label = "Uncertain 🤔"

    return label, float(prob)
