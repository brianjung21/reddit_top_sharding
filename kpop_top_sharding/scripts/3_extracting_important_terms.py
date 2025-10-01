"""
This script processes a text dataset (K‑pop sector), cleaning and analyzing text with the goal of extracting the most
frequent and contextually important terms. It supports multilingual text (English + Korean), removes brand/alias
terms to avoid leakage, and constructs both term frequency and TF‑IDF statistics.

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
    # keep only English letters and Hangul (U+3131–U+318E, U+AC00–U+D7A3)
    text = re.sub(r"[^a-zA-Z\u3131-\u318E\uAC00-\uD7A3\s]", " ", text)
    text = re.sub(r"\s+", " ", text)               # normalize spaces
    return text.strip()

df["clean_text"] = df["text"].map(clean_text)
df = df[df["clean_text"].str.len() > 0].copy()

# 4. Stopwords
stop_words = set(stopwords.words("english"))

# Minimal Korean stopwords (function words / particles)
ko_stop = {
    "은","는","이","가","을","를","에","의","와","과","도","에서","으로","하고","이나","라면",
    "및","등","그리고","그러나","그래서","또한","이런","그런","저런","좀","매우","너무","보다","보다도"
}
stop_words |= ko_stop

# Remove brand/alias leakage (if alias file exists)
kw_csv = Path("../files/kpop_keywords_with_aliases.csv")
if kw_csv.exists():
    try:
        kw_df = pd.read_csv(kw_csv)
        # Collect all string tokens from likely columns
        cols = [c for c in kw_df.columns if kw_df[c].dtype == object]
        alias_terms = set()
        for c in cols:
            alias_terms |= {str(x).strip().lower() for x in kw_df[c].dropna().tolist() if str(x).strip()}
        # Split multi‑word aliases into tokens, add both whole alias and tokens
        expanded = set()
        for term in alias_terms:
            expanded.add(term)
            expanded |= set(term.split())
        stop_words |= expanded
    except Exception:
        pass

# Optional domain‑specific extras
custom_stop = {"brandname1", "brandname2"}
stop_words |= custom_stop
stop_words = list(stop_words)

# 5. CountVectorizer for frequency
count_vec = CountVectorizer(stop_words=stop_words, min_df=2, token_pattern=r"(?u)\b[A-Za-z\u3131-\u318E\uAC00-\uD7A3]{2,}\b")  # ignore words appearing <2 times
count_matrix = count_vec.fit_transform(df["clean_text"])
word_counts = count_matrix.sum(axis=0).A1
count_df = pd.DataFrame({
    "term": count_vec.get_feature_names_out(),
    "frequency": word_counts
}).sort_values("frequency", ascending=False)

# 6. TF-IDF for contextual importance
tfidf_vec = TfidfVectorizer(stop_words=stop_words, min_df=2, token_pattern=r"(?u)\b[A-Za-z\u3131-\u318E\uAC00-\uD7A3]{2,}\b")
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