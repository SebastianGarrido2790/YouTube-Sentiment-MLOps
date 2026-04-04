# 1. Create directories if they don't exist
New-Item -ItemType Directory -Force -Path "src/pipeline"
New-Item -ItemType Directory -Force -Path "src/components"
New-Item -ItemType Directory -Force -Path "src/tools"

# 2. Move files and rename
# Data Ingestion
if (Test-Path "src/data/download_dataset.py") {
    Move-Item -Path "src/data/download_dataset.py" -Destination "src/pipeline/stage_01_data_ingestion.py" -Force
}

# Feature Exploratory Tools
if (Test-Path "src/features/tfidf_vs_distilbert.py") {
    Move-Item -Path "src/features/tfidf_vs_distilbert.py" -Destination "src/tools/feature_comparison.py" -Force
}
if (Test-Path "src/features/tfidf_max_features.py") {
    Move-Item -Path "src/features/tfidf_max_features.py" -Destination "src/tools/feature_tuning.py" -Force
}
if (Test-Path "src/features/imbalance_tuning.py") {
    Move-Item -Path "src/features/imbalance_tuning.py" -Destination "src/tools/imbalance_tuning.py" -Force
}

# Helpers
if (Test-Path "src/features/helpers") {
    Get-ChildItem -Path "src/features/helpers\*.py" | Move-Item -Destination "src/components\" -Force
}
if (Test-Path "src/models/helpers") {
    Get-ChildItem -Path "src/models/helpers\*.py" | Move-Item -Destination "src/components\" -Force
}
if (Test-Path "src/models/absa_model.py") {
    Move-Item -Path "src/models/absa_model.py" -Destination "src/components\absa_model.py" -Force
}

# Pipelines
if (Test-Path "src/models/baseline_logistic.py") {
    Move-Item -Path "src/models/baseline_logistic.py" -Destination "src/pipeline/stage_04a_baseline_model.py" -Force
}
if (Test-Path "src/models/hyperparameter_tuning.py") {
    Move-Item -Path "src/models/hyperparameter_tuning.py" -Destination "src/pipeline/stage_04b_hyperparameter_tuning.py" -Force
}
if (Test-Path "src/models/distilbert_training.py") {
    Move-Item -Path "src/models/distilbert_training.py" -Destination "src/pipeline/stage_04c_distilbert_training.py" -Force
}
if (Test-Path "src/models/model_evaluation.py") {
    Move-Item -Path "src/models/model_evaluation.py" -Destination "src/pipeline/stage_05_model_evaluation.py" -Force
}
if (Test-Path "src/models/register_model.py") {
    Move-Item -Path "src/models/register_model.py" -Destination "src/pipeline/stage_06_register_model.py" -Force
}

# Remove old directories completely
Remove-Item -Path "src/data" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "src/features" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "src/models" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "features" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "models" -Recurse -Force -ErrorAction SilentlyContinue

