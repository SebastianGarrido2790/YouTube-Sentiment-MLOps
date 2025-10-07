# Trains Logistic Regression baseline on TF-IDF features with undersampling for imbalance.
# Logs experiment to MLflow; saves model to models/.
# Fixes: Increased max_iter/solver for convergence; proper label mapping for metrics.
# Run: uv run python src/models/baseline_logistic.py

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
from scipy.sparse import load_npz
from imblearn.under_sampling import RandomUnderSampler
import pickle
import os

# Set up MLflow (adjust URI for your EC2 instance)
# mlflow.set_tracking_uri("http://ec2-54-175-41-29.compute-1.amazonaws.com:5000/")
mlflow.set_experiment("Baseline Model - Logistic Regression TF-IDF")


def train_baseline(scale_dense: bool = False) -> None:
    """
    Train Logistic Regression with undersampling.
    """
    # Load features (TF-IDF + derived; assumes default from feature_engineering.py)
    X_train = load_npz(
        "../../models/features/X_train.npz"
    ).tocsr()  # Convert to CSR for slicing
    X_test = load_npz("../../models/features/X_test.npz").tocsr()
    y_train = np.load("../../models/features/y_train.npy")
    y_test = np.load("../../models/features/y_test.npy")

    # Load label encoder to map back to original labels
    with open("../../models/features/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    original_labels = le.classes_  # [-1, 0, 1]

    # Apply undersampling to train set only
    rus = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

    # Model params (updated for convergence)
    C = 1.0  # Inverse regularization
    max_iter = 2000  # Increased
    solver = "liblinear"  # Better for sparse/unscaled

    # End any active run to avoid conflicts
    mlflow.end_run()

    with mlflow.start_run():
        # Tags and description
        mlflow.set_tag(
            "mlflow.runName", "LogisticRegression_Baseline_TF-IDF_UnderSampling"
        )
        mlflow.set_tag("experiment_type", "baseline_modeling")
        mlflow.set_tag("model_type", "LogisticRegression")
        mlflow.set_tag(
            "description",
            "Baseline Logistic Regression on TF-IDF features with RandomUnderSampler for class imbalance handling",
        )
        # Log params
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("imbalance_method", "RandomUnderSampler")
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("solver", solver)
        mlflow.log_param("feature_type", "TF-IDF + derived")
        mlflow.log_param("train_samples_post_sampling", X_train_resampled.shape[0])

        # Train
        model = LogisticRegression(
            C=C, max_iter=max_iter, solver=solver, random_state=42
        )
        model.fit(X_train_resampled, y_train_resampled)

        # Predict on test
        y_pred_encoded = model.predict(X_test)
        y_pred = le.inverse_transform(y_pred_encoded)
        y_test_original = le.inverse_transform(y_test)

        # Metrics (use original labels for interpretability)
        accuracy = accuracy_score(y_test_original, y_pred)
        macro_f1 = f1_score(y_test_original, y_pred, average="macro")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("macro_f1", macro_f1)

        # Per-class F1 (now with original labels)
        report = classification_report(y_test_original, y_pred, output_dict=True)
        for label in original_labels:
            mlflow.log_metric(f"{label}_f1", report[str(label)]["f1-score"])

        # Log model with encoder
        model_bundle = {"model": model, "encoder": le}
        mlflow.sklearn.log_model(model_bundle, "logistic_regression_baseline")

        print(f"Baseline trained: Accuracy {accuracy:.3f}, Macro F1 {macro_f1:.3f}")
        print(
            "Per-class F1:",
            {str(l): report[str(l)]["f1-score"] for l in original_labels},
        )

    # Save locally for inference
    local_dir = os.path.abspath("models/baseline")
    os.makedirs(local_dir, exist_ok=True)
    full_path = os.path.join(local_dir, "logistic_baseline.pkl")

    with open(full_path, "wb") as f:
        pickle.dump({"model": model, "encoder": le}, f)
        print(f"Model successfully saved to: {full_path}")


if __name__ == "__main__":
    train_baseline()
