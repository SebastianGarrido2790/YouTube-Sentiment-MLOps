# This script compares TF-IDF vs. BERT embeddings for sentiment vectorization using MLflow tracking.
# It trains RandomForest models on each, logs metrics/models, and saves visualizations.
# Assumes prepared data in data/processed/; MLflow server at provided URI.
# Run: uv run python src/features/01_tfidf_vs_bert.py

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.sparse import load_npz
import pickle
from typing import Tuple

# Set up MLflow (adjust URI for your EC2 instance)
# mlflow.set_tracking_uri("http://ec2-54-175-41-29.compute-1.amazonaws.com:5000/")
mlflow.set_experiment("Exp - TFIDF vs BERT")

# Ensure reports directory
os.makedirs("reports/figures/tfidf_vs_bert", exist_ok=True)


def get_bert_embeddings(
    texts: list, device: str = None, batch_size: int = 32
) -> np.ndarray:
    """
    Generate mean-pooled BERT embeddings for a list of texts.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(device)
            outputs = model(**inputs)
            # Mean pool over tokens (exclude CLS/SEP)
            pooled = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)
            embeddings.append(pooled.cpu().numpy())

    return np.vstack(embeddings)


def run_experiment(
    vectorizer_type: str,
    ngram_range: Tuple[int, int] = None,
    max_features: int = 5000,
    vectorizer_name: str = "TF-IDF",
) -> None:
    """
    Run experiment for given vectorizer type.

    Args:
        vectorizer_type: "TF-IDF" or "BERT".
        ngram_range: For TF-IDF only.
        max_features: For TF-IDF only.
        vectorizer_name: Display name.
    """
    # Load pre-split data (use train for training, test for eval; assume pre-cleaned)
    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")

    X_train_text = train_df["clean_comment"].tolist()
    y_train = train_df["category"].values
    X_test_text = test_df["clean_comment"].tolist()
    y_test = test_df["category"].values

    # Vectorization
    if vectorizer_type == "TF-IDF":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            stop_words="english",
            lowercase=True,
            min_df=2,
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)
        feature_dim = X_train.shape[1]
    elif vectorizer_type == "BERT":
        print("Generating BERT embeddings...")
        X_train = get_bert_embeddings(X_train_text)
        X_test = get_bert_embeddings(X_test_text)
        feature_dim = X_train.shape[1]  # 768
        ngram_range = (1, 1)  # Dummy for logging
    else:
        raise ValueError("Unsupported vectorizer_type")

    # Model params
    n_estimators = 200
    max_depth = 15

    with mlflow.start_run():
        # Tags and description
        mlflow.set_tag(
            "mlflow.runName", f"{vectorizer_name}_{ngram_range}_{n_estimators}est"
        )
        mlflow.set_tag("experiment_type", "feature_engineering")
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag(
            "description",
            f"RandomForest with {vectorizer_name}, ngram_range={ngram_range}, features={feature_dim}",
        )

        # Log params
        mlflow.log_param("vectorizer_type", vectorizer_type)
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param(
            "max_features", max_features if vectorizer_type == "TF-IDF" else None
        )
        mlflow.log_param("feature_dim", feature_dim)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)

        # Predict and metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Classification report
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in classification_rep.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix: {vectorizer_name}, {ngram_range}")
        plot_path = f"reports/figures/tfidf_vs_bert/confusion_matrix_{vectorizer_name}_{ngram_range}.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        # Log model
        mlflow.sklearn.log_model(
            model, f"random_forest_model_{vectorizer_name}_{ngram_range}"
        )


if __name__ == "__main__":
    # TF-IDF experiments with different n-grams
    ngram_ranges = [(1, 1), (1, 2), (1, 3)]
    max_features = 5000

    for ngram_range in ngram_ranges:
        run_experiment("TF-IDF", ngram_range, max_features, "TF-IDF")

    # BERT experiment (single run)
    run_experiment("BERT", (1, 1), max_features=768, vectorizer_name="BERT")
