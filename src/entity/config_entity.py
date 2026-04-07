"""
Pydantic schemas for project configuration.

This module defines strict data contracts for the application's configuration.
These schemas correspond directly to the structure of `params.yaml`.
Using Pydantic ensures that all configuration parameters are validated for type
and constraints at runtime, preventing silent failures due to misconfiguration.
"""

from pydantic import BaseModel, ConfigDict, Field


class DataIngestionConfig(BaseModel):
    """
    Configuration for the initial data download stage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(description="URL to the raw dataset (e.g., CSV on GitHub).")
    output_path: str = Field(description="Local path where the raw file will be saved.")


class DataPreparationConfig(BaseModel):
    """
    Configuration for data splitting and cleaning.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

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

    model_config = ConfigDict(frozen=True, extra="forbid")

    mlflow_uri: str = Field(description="URI for the MLflow tracking server.")
    ngram_ranges: list[list[int]] = Field(description="List of n-gram ranges to test (e.g., [[1,1], [1,2]]).")
    max_features: int = Field(description="Maximum vocabulary size for TF-IDF.")
    use_distilbert: bool = Field(description="Flag to enable/disable DistilBERT embeddings.")
    batch_size: int = Field(description="Batch size for deep learning inference.")
    n_estimators: int = Field(description="Trees in the Random Forest baseline.")
    max_depth: int = Field(description="Max depth for the Random Forest baseline.")


class FeatureTuningConfig(BaseModel):
    """
    Configuration for tuning TF-IDF max_features.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_features_values: list[int] = Field(description="List of max_features candidates to evaluate.")
    best_ngram_range: list[int] = Field(description="The chosen best performing n-gram range from the previous stage.")
    n_estimators: int = Field(description="Trees in the Random Forest baseline.")
    max_depth: int = Field(description="Max depth for the Random Forest baseline.")


class ImbalanceTuningConfig(BaseModel):
    """
    Configuration for handling class imbalance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    imbalance_methods: list[str] = Field(description="List of methods to test (e.g., ['smote', 'adasyn']).")
    best_max_features: int = Field(description="Best max_features from tuning stage.")
    best_ngram_range: list[int] = Field(description="Best n-gram range from comparison stage.")
    rf_n_estimators: int
    rf_max_depth: int


class FeatureEngineeringConfig(BaseModel):
    """
    Configuration for the final feature engineering pipeline.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    use_distilbert: bool = Field(description="Bool flag to select final strategy.")
    distilbert_batch_size: int
    best_max_features: int | None = None
    best_ngram_range: str | None = None


class LogisticBaselineConfig(BaseModel):
    """
    Hyperparameters for the Logistic Regression baseline.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_type: str
    class_weight: str
    solver: str
    max_iter: int
    C: float = Field(default=1.0, description="Inverse regularization strength.")


class LightGBMConfig(BaseModel):
    """
    Hyperparameters for LightGBM tuning.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_trials: int = Field(description="Number of Optuna trials.")


class XGBoostConfig(BaseModel):
    """
    Hyperparameters for XGBoost tuning.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_trials: int = Field(description="Number of Optuna trials.")


class DistilBERTConfig(BaseModel):
    """
    Hyperparameters for DistilBERT fine-tuning.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enable: bool
    n_trials: int
    batch_size: list[int]
    lr: list[float]
    weight_decay: list[float]


class HyperparameterTuningConfig(BaseModel):
    """
    Grouping for all model tuning configurations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lightgbm: LightGBMConfig
    xgboost: XGBoostConfig


class TrainConfig(BaseModel):
    """
    Master configuration for the training stage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    logistic_baseline: LogisticBaselineConfig
    hyperparameter_tuning: HyperparameterTuningConfig
    distilbert: DistilBERTConfig


class ModelEvaluationConfig(BaseModel):
    """
    Configuration for model evaluation metrics.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    models: list[str] = Field(description="List of model names to evaluate.")


class RegisterConfig(BaseModel):
    """
    Criteria for registering a model to production.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    f1_threshold: float = Field(description="Minimum F1 score required for registration.")


class AppConfig(BaseModel):
    """
    Root configuration object mapping the entire `params.yaml` structure.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_ingestion: DataIngestionConfig
    data_preparation: DataPreparationConfig
    feature_comparison: FeatureComparisonConfig
    feature_tuning: FeatureTuningConfig
    imbalance_tuning: ImbalanceTuningConfig
    feature_engineering: FeatureEngineeringConfig
    train: TrainConfig
    model_evaluation: ModelEvaluationConfig
    register_config: RegisterConfig = Field(alias="register")


class DataPathsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    raw_dir: str
    external_dir: str
    processed_dir: str
    raw_path: str
    train_path: str
    test_path: str
    val_path: str


class ModelsPathsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    root_dir: str
    baseline_dir: str
    advanced_dir: str
    features_dir: str
    evaluation_dir: str


class ReportsPathsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    root_dir: str
    figures_dir: str
    docs_dir: str
    eval_fig_dir: str
    tfidf_fig_dir: str
    imbalance_fig_dir: str


class OpsPathsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    logs_dir: str
    mlruns_dir: str
    gx_dir: str


class SystemConfig(BaseModel):
    """System Paths configuration (config.yaml)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    artifacts_root: str
    data: DataPathsConfig
    models: ModelsPathsConfig
    reports: ReportsPathsConfig
    ops: OpsPathsConfig


class SchemaConfig(BaseModel):
    """Data Contract configuration (schema.yaml)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    columns: dict[str, str]
    target: str
