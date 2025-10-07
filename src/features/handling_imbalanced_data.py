# This script experiments with various imbalance handling techniques using MLflow tracking.
# It trains RandomForest models on TF-IDF features (trigrams, max_features=10000), applies techniques to train set only,
# logs metrics/models, and saves visualizations. Uses preprocessed train/test splits for pipeline consistency.
# Run: uv run python src/features/04_handling_imbalanced_data.py

import mlflow
import mlflow.sklearn
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Set up MLflow (adjust URI for your EC2 instance)
# mlflow.set_tracking_uri("http://ec2-54-175-41-29.compute-1.amazonaws.com:5000/")
mlflow.set_experiment("Exp - Handling Imbalanced Data")

# Ensure reports directory
os.makedirs("reports/figures/imbalanced_data", exist_ok=True)


def run_imbalanced_experiment(imbalance_method: str) -> None:
    """
    Run experiment for specified imbalance handling method.

    Args:
        imbalance_method: Technique name ('class_weights', 'oversampling', 'adasyn', 'undersampling', 'smote_enn').
    """
    ngram_range = (1, 3)  # Trigram setting
    max_features = 10000  # Set max_features to 10000 for TF-IDF

    # Load pre-split data (train for training/resampling, test for eval; already cleaned)
    train_df = pd.read_parquet("data/processed/train.parquet")
    test_df = pd.read_parquet("data/processed/test.parquet")

    X_train_text = train_df["clean_comment"].tolist()
    y_train = train_df["category"].values
    X_test_text = test_df["clean_comment"].tolist()
    y_test = test_df["category"].values

    # Vectorization using TF-IDF, fit on training data only
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words="english",
        lowercase=True,
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    # Handle class imbalance based on the selected method (only applied to the training set)
    if imbalance_method == "class_weights":
        # Use class_weight in Random Forest
        class_weight = "balanced"
        # No resampling
    else:
        class_weight = None  # Do not apply class_weight if using resampling

        # Resampling Techniques (only apply to the training set)
        if imbalance_method == "oversampling":
            smote = SMOTE(random_state=42)
            X_train_vec, y_train = smote.fit_resample(X_train_vec, y_train)
        elif imbalance_method == "adasyn":
            adasyn = ADASYN(random_state=42)
            X_train_vec, y_train = adasyn.fit_resample(X_train_vec, y_train)
        elif imbalance_method == "undersampling":
            rus = RandomUnderSampler(random_state=42)
            X_train_vec, y_train = rus.fit_resample(X_train_vec, y_train)
        elif imbalance_method == "smote_enn":
            smote_enn = SMOTEENN(random_state=42)
            X_train_vec, y_train = smote_enn.fit_resample(X_train_vec, y_train)

    # Model params
    n_estimators = 200
    max_depth = 15

    with mlflow.start_run():
        # Tags and description
        mlflow.set_tag(
            "mlflow.runName",
            f"Imbalance_{imbalance_method}_RandomForest_TFIDF_Trigrams",
        )
        mlflow.set_tag("experiment_type", "imbalance_handling")
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag(
            "description",
            f"RandomForest with TF-IDF Trigrams, imbalance handling method={imbalance_method}",
        )

        # Log params
        mlflow.log_param("vectorizer_type", "TF-IDF")
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("vectorizer_max_features", max_features)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("imbalance_method", imbalance_method)

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight=class_weight,
        )
        model.fit(X_train_vec, y_train)

        # Predict and metrics
        y_pred = model.predict(X_test_vec)
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
        plt.title(f"Confusion Matrix: TF-IDF Trigrams, Imbalance={imbalance_method}")
        confusion_matrix_filename = (
            f"reports/figures/imbalanced_data/confusion_matrix_{imbalance_method}.png"
        )
        plt.savefig(confusion_matrix_filename)
        mlflow.log_artifact(confusion_matrix_filename)
        plt.close()

        # Log model
        mlflow.sklearn.log_model(
            model, f"random_forest_model_tfidf_trigrams_imbalance_{imbalance_method}"
        )


if __name__ == "__main__":
    # Run experiments for different imbalance methods
    imbalance_methods = [
        "class_weights",
        "oversampling",
        "adasyn",
        "undersampling",
        "smote_enn",
    ]
    for method in imbalance_methods:
        run_imbalanced_experiment(method)
