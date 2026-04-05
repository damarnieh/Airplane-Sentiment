# Airplane Passenger Tweet Sentiment Analysis

## Project Overview

This project focuses on analyzing customer opinions about airlines by performing **sentiment analysis on Twitter data**. The objective is to classify tweets related to airline services into **positive, neutral, or negative sentiments**.

By applying Natural Language Processing (NLP) techniques and machine learning models, this project demonstrates how textual data from social media can be transformed into meaningful insights about customer satisfaction and public perception of airline services.

---

## Problem Background

Airlines receive thousands of comments, complaints, and feedback from customers through social media platforms such as Twitter. These messages contain valuable information regarding customer experiences, including flight delays, customer service quality, baggage issues, and overall satisfaction.

Manually reviewing large volumes of tweets is inefficient and time-consuming. Therefore, sentiment analysis using machine learning provides an automated approach to categorize customer feedback and extract insights that can help improve airline services.

---

## Dataset

The dataset used in this project is the **Twitter US Airline Sentiment dataset**, which contains tweets mentioning major U.S. airlines.

Dataset characteristics:

* Approximately **14,000 tweets** collected from Twitter
* Text-based dataset
* Sentiment labels:

  * Positive
  * Neutral
  * Negative
* Additional attributes such as airline name, tweet metadata, and reasons for negative sentiment

---

## Project Workflow

### 1. Data Exploration

Initial data exploration was conducted to understand the dataset structure, sentiment distribution, and commonly appearing words.

### 2. Text Preprocessing

Text data was cleaned and prepared before modeling. The preprocessing steps include:

* Converting text to lowercase
* Removing punctuation and special characters
* Removing stopwords
* Tokenization
* Text normalization

### 3. Feature Engineering

The text data was converted into numerical representations using vectorization techniques such as:

* Bag of Words (CountVectorizer)
* TF-IDF (Term Frequency–Inverse Document Frequency)

### 4. Model Training

Machine learning algorithms were used to classify tweet sentiments. The models were trained using the processed text features.

### 5. Model Evaluation

Model performance was evaluated using classification metrics such as:

* Accuracy
* Precision
* Recall
* F1-score

---

## Tech Stack

Programming Language:

* Python

Libraries:

* pandas
* numpy
* scikit-learn
* nltk
* matplotlib
* seaborn

Tools:

* Jupyter Notebook

---

## Project Outcome

The trained model can automatically classify airline-related tweets into sentiment categories. This approach enables companies to quickly analyze public opinion and identify major issues raised by customers.

Such analysis can help airlines:

* Monitor customer satisfaction
* Detect recurring service issues
* Improve customer experience and service quality

---

## Conclusion

Sentiment analysis is an effective method for analyzing large volumes of social media data. By applying NLP techniques and machine learning models, this project demonstrates how textual data can be transformed into meaningful insights about customer perception toward airline services.
