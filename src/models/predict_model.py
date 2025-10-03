import pickle
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


def predict_sentiment(
    comments: list[str],
    model_path: str = "models/logistic_baseline.pkl",
    vectorizer_path: str = "data/processed/features/vectorizer.pkl",
) -> list[str]:
    """
    Predict sentiment for new comments (-1: Negative, 0: Neutral, 1: Positive).

    Args:
        comments: List of raw comments.

    Returns:
        List of predicted labels.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open("data/processed/features/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    # Clean and vectorize (reuse clean_text from src/data/make_dataset.py)
    from src.data.make_dataset import clean_text

    cleaned = [clean_text(c) for c in comments]
    X_new = vectorizer.transform(cleaned)

    # Derived features (simplified; match engineering)
    df_temp = pd.DataFrame({"clean_comment": cleaned})
    df_temp["char_len"] = df_temp["clean_comment"].str.len()
    df_temp["word_len"] = df_temp["clean_comment"].str.split().str.len()
    pos_words = {"good", "great", "love", "like", "positive", "best"}
    neg_words = {"bad", "hate", "worst", "negative", "shit", "fuck"}
    df_temp["pos_ratio"] = df_temp["clean_comment"].apply(
        lambda x: len([w for w in x.split() if w in pos_words]) / max(len(x.split()), 1)
    )
    df_temp["neg_ratio"] = df_temp["clean_comment"].apply(
        lambda x: len([w for w in x.split() if w in neg_words]) / max(len(x.split()), 1)
    )
    derived = df_temp[["char_len", "word_len", "pos_ratio", "neg_ratio"]].values
    X_final = hstack([X_new, derived])

    y_pred = model.predict(X_final)
    labels = le.inverse_transform(y_pred)
    return [{-1: "Negative", 0: "Neutral", 1: "Positive"}[l] for l in labels]


# Example
if __name__ == "__main__":
    sample_comments = ["This video is amazing!", "Terrible content, waste of time."]
    preds = predict_sentiment(sample_comments)
    print(preds)
