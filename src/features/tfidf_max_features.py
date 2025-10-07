# This script experiments with varying max_features for TF-IDF (trigrams) using MLflow tracking.
# It trains RandomForest models on each configuration, logs metrics/models, and saves visualizations.
# Uses preprocessed train/test splits for consistency with the pipeline.
# Run: uv run python src/features/02_tfidf_max_features.py

import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Set up MLflow (adjust URI for your EC2 instance)
# mlflow.set_tracking_uri("http://ec2-54-175-41-29.compute-1.amazonaws.com:5000/")
mlflow.set_experiment("Exp - TFIDF max_features")

# Ensure reports directory
os.makedirs("reports/figures/tfidf_max_features", exist_ok=True)


def run_experiment_tfidf_max_features(max_features: int) -> None:
    """
    Run experiment for TF-IDF with specified max_features (trigrams).

    Args:
        max_features: Maximum number of features for TF-IDF.
    """
    ngram_range = (1, 3)  # Trigram setting

    # Load pre-split data (train for training, test for eval; already cleaned)
    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")

    X_train_text = train_df["clean_comment"].tolist()
    y_train = train_df["category"].values
    X_test_text = test_df["clean_comment"].tolist()
    y_test = test_df["category"].values

    # Vectorization using TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words="english",
        lowercase=True,
        min_df=2,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Model params
    n_estimators = 200
    max_depth = 15

    with mlflow.start_run():
        # Tags and description
        mlflow.set_tag("mlflow.runName", f"TFIDF_Trigrams_max_features_{max_features}")
        mlflow.set_tag("experiment_type", "feature_engineering")
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag(
            "description",
            f"RandomForest with TF-IDF Trigrams, max_features={max_features}",
        )

        # Log params
        mlflow.log_param("vectorizer_type", "TF-IDF")
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("vectorizer_max_features", max_features)
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
        plt.title(f"Confusion Matrix: TF-IDF Trigrams, max_features={max_features}")
        plot_path = f"reports/figures/tfidf_max_features/confusion_matrix_tfidf_max_{max_features}.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        # Log model
        mlflow.sklearn.log_model(
            model, f"random_forest_model_tfidf_trigrams_{max_features}"
        )


if __name__ == "__main__":
    # Test various max_features values
    max_features_values = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    for max_features in max_features_values:
        run_experiment_tfidf_max_features(max_features)
