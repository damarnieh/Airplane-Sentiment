import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter



# NLTK SETUP (HF SAFE)
NLTK_RESOURCES = [
    "punkt",
    "punkt_tab",
    "stopwords"
]

for res in NLTK_RESOURCES:
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download(res)

stop_words = set(stopwords.words("english"))


def run():
    st.title("Exploratory Data Analysis")
    st.markdown("Exploratory analysis of airline tweet sentiment dataset")

    # LOAD DATA
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(
        BASE_DIR,
        "data",
        "Tweets.csv"
    )

    df = pd.read_csv(DATA_PATH)

    st.write(f"Total records: {df.shape[0]} rows")
    st.write(f"Total features: {df.shape[1]} columns")

    st.dataframe(df.head())

    
    # TARGET DISTRIBUTION
    st.markdown("## Sentiment Distribution")

    fig, ax = plt.subplots()
    sns.countplot(
        x="airline_sentiment",
        data=df,
        order=df["airline_sentiment"].value_counts().index,
        ax=ax
    )
    ax.set_title("Sentiment Class Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.markdown(
        """
        The target variable is highly imbalanced.
        Negative sentiment dominates the dataset, which may cause bias during model training
        if not handled properly.
        """
    )

    
    # TEXT LENGTH ANALYSIS
    st.markdown("## Tweet Length Analysis")

    df["text_length"] = df["text"].astype(str).apply(len)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        x="airline_sentiment",
        y="text_length",
        data=df,
        ax=ax
    )
    ax.set_title("Tweet Length by Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Characters")
    st.pyplot(fig)

    st.markdown(
        """
        Tweets with negative sentiment tend to have slightly longer text,
        indicating that users often provide more detailed explanations when complaining.
        """
    )

    
    # WORD COUNT DISTRIBUTION
    st.markdown("## Word Count Distribution")

    df["word_count"] = df["text"].astype(str).apply(lambda x: len(x.split()))

    fig, ax = plt.subplots()
    sns.histplot(df["word_count"], bins=30, kde=True, ax=ax)
    ax.set_title("Word Count Distribution")
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    
    # BASIC TEXT CLEANING (EDA)
    def clean_text(text):
        if not isinstance(text, str):
            return []

        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))

        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
        return tokens

    df["clean_tokens"] = df["text"].apply(clean_text)

    
    # MOST COMMON WORDS
    st.markdown("## Most Common Words per Sentiment")

    def get_top_words(sentiment, n=10):
        words = []
        subset = df[df["airline_sentiment"] == sentiment]
        for tokens in subset["clean_tokens"]:
            words.extend(tokens)
        return Counter(words).most_common(n)

    for sentiment in df["airline_sentiment"].unique():
        st.subheader(f"Top Words for {sentiment.capitalize()} Sentiment")
        top_words = get_top_words(sentiment)

        word_df = pd.DataFrame(top_words, columns=["Word", "Frequency"])
        st.dataframe(word_df)

    
    # AIRLINE VS SENTIMENT
    st.markdown("## Sentiment Distribution per Airline")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(
        x="airline",
        hue="airline_sentiment",
        data=df,
        ax=ax
    )
    ax.set_title("Sentiment per Airline")
    ax.set_xlabel("Airline")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.markdown(
        """
        Different airlines show varying sentiment distributions.
        Some airlines receive a significantly higher proportion of negative feedback,
        indicating potential service quality issues.
        """
    )

    
    # MISSING VALUE CHECK
    st.markdown("## Missing Value Check")
    st.write(df.isnull().sum())

    
    # EDA CONCLUSION
    st.markdown("## EDA Summary")

    st.markdown(
        """
        - The dataset is dominated by negative sentiment, indicating strong class imbalance.
        - Tweet length and word count vary across sentiment categories.
        - Negative tweets often contain complaint-related vocabulary.
        - Sentiment distribution differs significantly between airlines.
        - The dataset is suitable for sentiment classification with proper preprocessing
          and class balancing strategies.
        """
    )


if __name__ == "__main__":
    run()
