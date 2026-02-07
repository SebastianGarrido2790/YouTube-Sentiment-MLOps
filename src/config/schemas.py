"""
Pydantic schemas for project configuration.

This module defines strict data contracts for the application's configuration.
These schemas correspond directly to the structure of `params.yaml`.
Using Pydantic ensures that all configuration parameters are validated for type
and constraints at runtime, preventing silent failures due to misconfiguration.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DataIngestionConfig(BaseModel):
    """
    Configuration for the initial data download stage.
    """

    url: str = Field(description="URL to the raw dataset (e.g., CSV on GitHub).")
    output_path: str = Field(description="Local path where the raw file will be saved.")


class DataPreparationConfig(BaseModel):
    """
    Configuration for data splitting and cleaning.
    """

    test_size: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Proportion of dataset to use for testing.",
    )
    random_state: int = Field(default=42, description="Seed for reproducibility.")


class FeatureComparisonConfig(BaseModel):
    """
    Configuration for the TF-IDF vs DistilBERT experiment stage.
    """

    mlflow_uri: str = Field(description="URI for the MLflow tracking server.")
    ngram_ranges: List[List[int]] = Field(
        description="List of n-gram ranges to test (e.g., [[1,1], [1,2]])."
    )
    max_features: int = Field(description="Maximum vocabulary size for TF-IDF.")
    use_distilbert: bool = Field(
        description="Flag to enable/disable DistilBERT embeddings."
    )
    batch_size: int = Field(description="Batch size for deep learning inference.")
    n_estimators: int = Field(description="Trees in the Random Forest baseline.")
    max_depth: int = Field(description="Max depth for the Random Forest baseline.")


class FeatureTuningConfig(BaseModel):
    """
    Configuration for tuning TF-IDF max_features.
    """

    max_features_values: List[int] = Field(
        description="List of max_features candidates to evaluate."
    )
    best_ngram_range: List[int] = Field(
        description="The chosen best performing n-gram range from the previous stage."
    )
    n_estimators: int = Field(description="Trees in the Random Forest baseline.")
    max_depth: int = Field(description="Max depth for the Random Forest baseline.")


class ImbalanceTuningConfig(BaseModel):
    """
    Configuration for handling class imbalance.
    """

    imbalance_methods: str = Field(
        description='String representation of list of methods to test (e.g., \'["smote", "adasyn"]\').'
    )
    best_max_features: int = Field(description="Best max_features from tuning stage.")
    best_ngram_range: str = Field(
        description="Best n-gram range from comparison stage."
    )
    rf_n_estimators: int
    rf_max_depth: int


class FeatureEngineeringConfig(BaseModel):
    """
    Configuration for the final feature engineering pipeline.
    """

    use_distilbert: str = Field(
        description="String boolean ('True'/'False') to select final strategy."
    )
    distilbert_batch_size: int
    best_max_features: Optional[int] = None
    best_ngram_range: Optional[str] = None


class LogisticBaselineConfig(BaseModel):
    """
    Hyperparameters for the Logistic Regression baseline.
    """

    model_type: str
    class_weight: str
    solver: str
    max_iter: int


class LightGBMConfig(BaseModel):
    """
    Hyperparameters for LightGBM tuning.
    """

    n_trials: int = Field(description="Number of Optuna trials.")


class XGBoostConfig(BaseModel):
    """
    Hyperparameters for XGBoost tuning.
    """

    n_trials: int = Field(description="Number of Optuna trials.")


class DistilBERTConfig(BaseModel):
    """
    Hyperparameters for DistilBERT fine-tuning.
    """

    enable: bool
    n_trials: int
    batch_size: List[int]
    lr: List[float]
    weight_decay: List[float]


class HyperparameterTuningConfig(BaseModel):
    """
    Grouping for all model tuning configurations.
    """

    lightgbm: LightGBMConfig
    xgboost: XGBoostConfig


class TrainConfig(BaseModel):
    """
    Master configuration for the training stage.
    """

    logistic_baseline: LogisticBaselineConfig
    hyperparameter_tuning: HyperparameterTuningConfig
    distilbert: DistilBERTConfig


class ModelEvaluationConfig(BaseModel):
    """
    Configuration for model evaluation metrics.
    """

    models: List[str] = Field(description="List of model names to evaluate.")


class RegisterConfig(BaseModel):
    """
    Criteria for registering a model to production.
    """

    f1_threshold: float = Field(
        description="Minimum F1 score required for registration."
    )


class AppConfig(BaseModel):
    """
    Root configuration object mapping the entire `params.yaml` structure.
    """

    data_ingestion: DataIngestionConfig
    data_preparation: DataPreparationConfig
    feature_comparison: FeatureComparisonConfig
    feature_tuning: FeatureTuningConfig
    imbalance_tuning: ImbalanceTuningConfig
    feature_engineering: FeatureEngineeringConfig
    train: TrainConfig
    model_evaluation: ModelEvaluationConfig
    register_config: RegisterConfig = Field(alias="register")
