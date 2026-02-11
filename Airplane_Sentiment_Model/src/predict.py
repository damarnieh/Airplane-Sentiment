import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences




# NLTK SETUP
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))




# PATH SETUP
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model_files",
    "best_ann_sentiment_model.keras"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "model_files",
    "tokenizer.pkl"
)




# LOAD MODEL & TOKENIZER
model = load_model(MODEL_PATH, compile=False)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 50   # ⚠️ HARUS SAMA DENGAN TRAINING




# TEXT PREPROCESSING
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = word_tokenize(text)
    tokens = [
        w for w in tokens
        if w.isalpha() and w not in stop_words
    ]

    return " ".join(tokens)




# STREAMLIT APP
def run():

    st.title("Sentiment Analysis Classification")
    st.markdown(
        "This application classifies **sentiment** "
        "(Negative / Neutral / Positive) from an input sentence."
    )

    with st.form("sentiment_form"):
        user_text = st.text_area(
            "Input Text",
            placeholder="Example: This flight was amazing and the crew was very helpful"
        )
        submit = st.form_submit_button("Classify")

    if submit:

        if user_text.strip() == "":
            st.warning("Please enter some text.")
            return

        # preprocessing
        cleaned_text = clean_text(user_text)

        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded_seq = pad_sequences(
            seq,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

        # prediction
        probs = model.predict(padded_seq)[0]
        pred_class = int(np.argmax(probs))

        label_map = {
            0: "NEGATIVE",
            1: "NEUTRAL",
            2: "POSITIVE"
        }

        sentiment = label_map[pred_class]

        # DISPLAY INPUT
        st.subheader("Input Data")
        st.dataframe(pd.DataFrame({
            "Original Text": [user_text],
            "Cleaned Text": [cleaned_text]
        }))

        # RESULT
        st.subheader("Classification Result")

        if sentiment == "POSITIVE":
            st.success(f"Sentiment: **{sentiment}**")
        elif sentiment == "NEUTRAL":
            st.info(f"Sentiment: **{sentiment}**")
        else:
            st.error(f"Sentiment: **{sentiment}**")

        st.write("Prediction Probabilities:")
        st.write({
            "Negative": float(probs[0]),
            "Neutral": float(probs[1]),
            "Positive": float(probs[2]),
        })


if __name__ == "__main__":
    run()
