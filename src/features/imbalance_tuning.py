"""
Experiment with imbalance handling techniques for sentiment classification.

Applies techniques (SMOTE, ADASYN, etc.) to TF-IDF features on the train set only, trains RandomForest,
and logs to MLflow using reusable helper functions.

Usage:
    uv run python -m src.features.imbalance_tuning --imbalance_methods "['class_weights','oversampling']" --max_features 1000

Requirements:
    - Processed data in data/processed/.
    - uv sync (for imblearn, scikit-learn, mlflow).
    - MLflow server running (e.g., uv run mlflow server --host 127.0.0.1 --port 5000).

Design Considerations:
- Reliability: Train-only resampling; validation on untouched test.
- Scalability: Sparse matrices; efficient resampling.
- Maintainability: Logging, type hints, relative paths.
- Adaptability: Parameterized methods; extensible to other classifiers.
"""

import argparse
from typing import Tuple, Dict, Any, Union
import mlflow
import numpy as np
from scipy.sparse import spmatrix  # For sparse matrix type hint
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Project Utilities ---
from src.utils.paths import FIGURES_DIR
from src.utils.logger import get_logger
from src.features.helpers.feature_utils import (
    setup_mlflow_run,
    load_train_val_data,
    parse_dvc_param,
    evaluate_and_log,
)

# --- Logging Setup ---
logger = get_logger(__name__, headline="imbalance_tuning.py")

# --- Path setup ---
IMBALANCE_FIGURES_DIR = FIGURES_DIR / "imbalance_methods"
IMBALANCE_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_imbalanced_experiment(
    imbalance_method: str,
    ngram_range: Tuple[int, int],
    max_features: int,
    n_estimators: int,
    max_depth: int,
) -> None:
    """
    Run experiment for specified imbalance handling method.

    Args:
        imbalance_method: Technique name ('class_weights', 'oversampling', 'adasyn', 'undersampling', 'smote_enn').
        ngram_range: N-gram range tuple.
        max_features: Maximum TF-IDF features.
        n_estimators: RF trees.
        max_depth: RF depth.
    """

    # --- Load data ---
    # NOTE: load_train_val_data loads train.parquet and val.parquet for hyperparameter tuning.
    train_df, val_df = load_train_val_data()

    X_train_text = train_df["clean_comment"].tolist()
    # Assuming 'category_encoded' is the numerical target variable (0, 1, 2)
    y_train = train_df["category_encoded"].values
    X_val_text = val_df["clean_comment"].tolist()
    y_val = val_df["category_encoded"].values

    logger.info(
        f"Data split: Train {train_df.shape[0]} ({np.bincount(y_train)}), Val {val_df.shape[0]}"
    )

    # --- Vectorization using TF-IDF (Fit on training, transform both) ---
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words="english",
        lowercase=False,
        min_df=2,
    )
    X_train_vec: spmatrix = vectorizer.fit_transform(X_train_text)
    X_val_vec: spmatrix = vectorizer.transform(X_val_text)
    feature_dim = X_train_vec.shape[1]

    # --- Handle class imbalance (train set only) ---
    class_weight: Union[str, None] = None
    resampling_applied: bool = False

    if imbalance_method == "class_weights":
        class_weight = "balanced"
        logger.info("Using 'balanced' class weights. No resampling applied.")
    else:
        sampler = None
        if imbalance_method == "oversampling":
            sampler = SMOTE(random_state=42)
        elif imbalance_method == "adasyn":
            sampler = ADASYN(random_state=42)
        elif imbalance_method == "undersampling":
            sampler = RandomUnderSampler(random_state=42)
        elif imbalance_method == "smote_enn":
            sampler = SMOTEENN(random_state=42, smote=SMOTE(random_state=42))
        else:
            raise ValueError(f"Unsupported imbalance method: {imbalance_method}")

        if sampler:
            resampling_applied = True
            logger.info(f"Applying {imbalance_method.upper()} to training data...")
            X_train_vec, y_train = sampler.fit_resample(X_train_vec, y_train)
            logger.info(
                f"New training sample size: {X_train_vec.shape[0]} (Classes: {np.bincount(y_train)})"
            )  # np.bincount for sanity check: count of samples per class after resampling

    # --- MLflow Tracking, Training, and Evaluation ---
    run_name = f"Imb_{imbalance_method}_Feat_{feature_dim}"
    logger.info(f"🚀 Running experiment: {run_name}")

    with mlflow.start_run(run_name=run_name):
        # 1. Model Training
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight=class_weight,
            n_jobs=-1,
        )
        model.fit(X_train_vec, y_train)

        # 2. Define Params and Tags for Logging
        params: Dict[str, Any] = {
            "vectorizer_type": "TF-IDF",
            "ngram_range": ngram_range,
            "max_features": max_features,
            "feature_dim": feature_dim,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "imbalance_method": imbalance_method,
            "class_weight_param": class_weight,
            "resampling_applied": resampling_applied,
        }
        tags: Dict[str, str] = {
            "experiment_type": "imbalance_tuning",
            "model_type": "RandomForestClassifier",
            "imbalance_method": imbalance_method,
        }

        # 3. Evaluation and Logging
        metrics = evaluate_and_log(
            model=model,
            X_val=X_val_vec,
            y_val=y_val,
            run_name=run_name,
            params=params,
            tags=tags,
            output_dir=IMBALANCE_FIGURES_DIR,
            log_model=True,  # Log the model to the MLflow registry
        )

        # 4. Log key metric to console
        # Use .get() to safely retrieve the metric, defaulting to 0.0 if the key is missing.
        val_accuracy = metrics.get("val_accuracy", 0.0)
        f1_score_1 = metrics.get("1_f1-score", 0.0)

        logger.info(f"Model Val Accuracy: {val_accuracy:.4f}")
        logger.info(f"Class 1 F1-Score: {f1_score_1:.4f}")

        logger.info(
            f"✅ Experiment finished: {run_name} | MLflow Run ID: {mlflow.last_active_run().info.run_id}"
        )


def main() -> None:
    """Parse args and run experiments."""
    parser = argparse.ArgumentParser(
        description="Handle class imbalance with MLflow tracking."
    )
    parser.add_argument(
        "--imbalance_methods",
        type=str,
        default="['class_weights','oversampling','adasyn','undersampling','smote_enn']",
        help='List of imbalance methods to test (e.g., \'["smote", "weights"]\').',
    )
    parser.add_argument(
        "--ngram_range", type=str, default="(1,1)", help="N-gram range as string tuple."
    )
    parser.add_argument(
        "--max_features", type=int, default=1000, help="Max TF-IDF features."
    )
    parser.add_argument(
        "--n_estimators", type=int, default=200, help="RF n_estimators."
    )
    parser.add_argument("--max_depth", type=int, default=15, help="RF max_depth.")
    args = parser.parse_args()

    # --- MLflow Setup ---
    setup_mlflow_run(experiment_name="Exp - Imbalance Handling")

    # --- Parameter Parsing ---
    imbalance_methods = parse_dvc_param(
        args.imbalance_methods, name="imbalance_methods", expected_type=list
    )
    ngram_range = parse_dvc_param(
        args.ngram_range, name="ngram_range", expected_type=tuple
    )
    # Note: parse_dvc_param handles the ast.literal_eval and validation internally.

    logger.info(
        f"--- Running imbalance experiments for methods: {imbalance_methods} ---"
    )
    for method in imbalance_methods:
        run_imbalanced_experiment(
            imbalance_method=method,
            ngram_range=ngram_range,
            max_features=args.max_features,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )
    logger.info(
        "--- Imbalance handling tuning complete. Analyze results in MLflow UI ---"
    )


if __name__ == "__main__":
    main()
