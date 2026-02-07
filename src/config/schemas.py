from typing import List, Optional
from pydantic import BaseModel, Field


class DataIngestionConfig(BaseModel):
    url: str
    output_path: str


class DataPreparationConfig(BaseModel):
    test_size: float = Field(default=0.15, ge=0.0, le=1.0)
    random_state: int = Field(default=42)


class FeatureComparisonConfig(BaseModel):
    mlflow_uri: str
    ngram_ranges: List[List[int]]
    max_features: int
    use_distilbert: bool
    batch_size: int
    n_estimators: int
    max_depth: int


class FeatureTuningConfig(BaseModel):
    max_features_values: str  # Stored as string in params.yaml
    best_ngram_range: str  # stored as tuple string
    n_estimators: int
    max_depth: int


class ImbalanceTuningConfig(BaseModel):
    imbalance_methods: str  # Stored as string repr of list
    best_max_features: int
    best_ngram_range: str
    rf_n_estimators: int
    rf_max_depth: int


class FeatureEngineeringConfig(BaseModel):
    use_distilbert: str  # Stored as string "False"/"True" in params
    distilbert_batch_size: int
    best_max_features: Optional[int] = None
    best_ngram_range: Optional[str] = None


class LogisticBaselineConfig(BaseModel):
    model_type: str
    class_weight: str
    solver: str
    max_iter: int


class LightGBMConfig(BaseModel):
    n_trials: int


class XGBoostConfig(BaseModel):
    n_trials: int


class DistilBERTConfig(BaseModel):
    enable: bool
    n_trials: int
    batch_size: List[int]
    lr: List[float]
    weight_decay: List[float]


class HyperparameterTuningConfig(BaseModel):
    lightgbm: LightGBMConfig
    xgboost: XGBoostConfig


class TrainConfig(BaseModel):
    logistic_baseline: LogisticBaselineConfig
    hyperparameter_tuning: HyperparameterTuningConfig
    distilbert: DistilBERTConfig


class ModelEvaluationConfig(BaseModel):
    models: List[str]


class RegisterConfig(BaseModel):
    f1_threshold: float


class AppConfig(BaseModel):
    data_ingestion: DataIngestionConfig
    data_preparation: DataPreparationConfig
    feature_comparison: FeatureComparisonConfig
    feature_tuning: FeatureTuningConfig
    imbalance_tuning: ImbalanceTuningConfig
    feature_engineering: FeatureEngineeringConfig
    train: TrainConfig
    model_evaluation: ModelEvaluationConfig
    register_config: RegisterConfig = Field(alias="register")
