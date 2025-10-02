# Trains Logistic Regression baseline on TF-IDF features with undersampling for imbalance.
# Logs experiment to MLflow; saves model to models/.
# Run: uv run python src/models/train_model.py

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.sparse import load_npz
from imblearn.under_sampling import RandomUnderSampler
import pickle
import os

# Set up MLflow (adjust URI for your EC2 instance)
# mlflow.set_tracking_uri("http://ec2-54-175-41-29.compute-1.amazonaws.com:5000/")
mlflow.set_experiment("Baseline - Logistic Regression TF-IDF")


def train_baseline() -> None:
    """
    Train Logistic Regression with undersampling.
    """
    # Load features (TF-IDF + derived; assumes default from feature_engineering.py)
    X_train = load_npz("model/features/X_train.npz")
    X_val = load_npz("models/features/X_val.npz")  # Unused for baseline
    X_test = load_npz("models/features/X_test.npz")
    y_train = np.load("models/features/y_train.npy")
    y_test = np.load("models/features/y_test.npy")

    # Apply undersampling to train set only
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    # Model params (baseline defaults; tune with Optuna later)
    C = 1.0  # Inverse regularization
    max_iter = 1000

    with mlflow.start_run():
        # Log params
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("imbalance_method", "RandomUnderSampler")
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("feature_type", "TF-IDF + derived")
        mlflow.log_param("train_samples_post_sampling", X_train_resampled.shape[0])

        # Train
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
        model.fit(X_train_resampled, y_train_resampled)

        # Predict on test
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("macro_f1", macro_f1)

        # Per-class F1
        report = classification_report(y_test, y_pred, output_dict=True)
        for label in ["-1", "0", "1"]:
            mlflow.log_metric(f"{label}_f1", report[label]["f1-score"])

        # Log model
        mlflow.sklearn.log_model(model, "logistic_regression_baseline")

        print(f"Baseline trained: Accuracy {accuracy:.3f}, Macro F1 {macro_f1:.3f}")

    # Save locally for inference prototyping
    os.makedirs("models", exist_ok=True)
    with open("models/logistic_baseline.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save scaler/encoder if needed (reuse from features)
    with open("data/processed/features/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)


if __name__ == "__main__":
    train_baseline()
