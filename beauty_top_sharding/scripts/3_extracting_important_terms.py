"""
This script processes a text dataset, cleaning and analyzing text with the goal of extracting the most
frequent and contextually important terms. It performs text cleaning, stopword removal, and constructs
both term frequency and TF-IDF statistics to identify key terms in the provided dataset.

The results of the analysis are saved to a CSV file.

Attributes
----------
DATA_PATH : Path
    Path to the input CSV dataset containing raw text data.
OUTPUT_PATH : Path
    Path to the output CSV file where the processed top terms with their frequencies and TF-IDF scores
    will be saved.
df : pandas.DataFrame
    The dataset read from the input file.
text_cols : list of str
    The list of text columns in the dataset to be concatenated for analysis.
stop_words : list of str
    The list of stopwords used for filtering out irrelevant or overly common terms.

Functions
---------
clean_text(text: str) -> str
    Processes a string of text, converting it to lowercase, removing URLs and non-alphabetic characters,
    and normalizing spacing.

Raises
------
Stopword download issue (nltk.download)
    If NLTK fails to download the stopwords, a runtime error may occur when using the stopword list.

"""

import pandas as pd
import re
import string
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Make sure stopwords are downloaded
nltk.download("stopwords")

# Paths
DATA_PATH = Path("../data/hot_samples.csv")
OUTPUT_PATH = Path("../data/sector_terms.csv")

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

# 2. Concatenate text columns into one
text_cols = ["title", "selftext", "comments_text"]
df["text"] = df[text_cols].fillna("").agg(" ".join, axis=1)

# 3. Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)           # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)       # keep only letters
    text = re.sub(r"\s+", " ", text)               # normalize spaces
    return text.strip()

df["clean_text"] = df["text"].map(clean_text)
df = df[df["clean_text"].str.len() > 0].copy()

# 4. Stopwords
stop_words = set(stopwords.words("english"))
# You can add domain-specific stopwords here, e.g. brand names:
custom_stop = {"brandname1", "brandname2"}
stop_words = stop_words.union(custom_stop)
stop_words = list(stop_words)

# 5. CountVectorizer for frequency
count_vec = CountVectorizer(stop_words=stop_words, min_df=2, token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")  # ignore words appearing <2 times
count_matrix = count_vec.fit_transform(df["clean_text"])
word_counts = count_matrix.sum(axis=0).A1
count_df = pd.DataFrame({
    "term": count_vec.get_feature_names_out(),
    "frequency": word_counts
}).sort_values("frequency", ascending=False)

# 6. TF-IDF for contextual importance
tfidf_vec = TfidfVectorizer(stop_words=stop_words, min_df=2, token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
tfidf_matrix = tfidf_vec.fit_transform(df["clean_text"])
tfidf_scores = tfidf_matrix.sum(axis=0).A1
tfidf_df = pd.DataFrame({
    "term": tfidf_vec.get_feature_names_out(),
    "tfidf_score": tfidf_scores
}).sort_values("tfidf_score", ascending=False)

# 7. Merge and save
merged = pd.merge(count_df, tfidf_df, on="term", how="outer").fillna(0)
merged = merged.sort_values("frequency", ascending=False)
merged.to_csv(OUTPUT_PATH, index=False)

print(f"Saved top terms to {OUTPUT_PATH.resolve()}")
print(merged.head(20))