# Test Suite Documentation

This document provides a comprehensive overview of the testing strategy for the YouTube Sentiment Analysis project. The test suite uses `pytest` and covers configuration management, data validation, and model pipeline orchestration.

## 🧪 Test Structure

The tests are organized into the following modules:

1.  **`tests/test_config.py`**: Validates the `ConfigurationManager` and strict Pydantic schemas.
2.  **`tests/test_data_validation.py`**: Verifies text cleaning logic in `src/data/make_dataset.py`, using parameterized tests for diverse scenarios.
3.  **`tests/test_model_pipeline.py`**: Tests the orchestration of the baseline model training using mocks.
4.  **`tests/conftest.py`**: Defines shared fixtures, including a mock `params.yaml` and module path setup.

---

## 1. Configuration Tests (`test_config.py`)

These tests ensure that strict typing and schema validation work as expected.

### `test_config_loading`
-   **Goal**: Ensure the global `AppConfig` object is correctly initialized.
-   **Method**: Uses the `config_manager` fixture (which loads a mock `params.yaml`) and verifies that returned objects are instances of the correct Pydantic models.

### `test_get_data_preparation_config`
-   **Goal**: Verify proper retrieval of nested configurations (e.g., `data_preparation`).
-   **Method**: Calls `get_data_preparation_config()` and asserts that the returned values matches the mock inputs.

### `test_missing_config_file`
-   **Goal**: Test error handling when configuration files are missing.
-   **Method**: Instantiates `ConfigurationManager` with an invalid path and asserts that it raises a `RuntimeError` when access is attempted.

---

## 2. Data Validation Tests (`test_data_validation.py`)

These tests focus on the `clean_text` function in `src/data/make_dataset.py`. We utilize `pytest.mark.parametrize` to efficiently test multiple input-output pairs.

### `test_clean_text_general`
-   **Goal**: Test general text cleaning scenarios including basic cleaning, special character removal, and edge case handling.
-   **Parametrized Cases**:
    -   `basic_cleaning`: "Hello World! 123" -> "hello world"
    -   `special_chars`: "Sentim@nt An@lys!s #" -> "sentim nt an lys s"
    -   `empty_string`: "" -> ""
    -   `pd_NA`: `pandas.NA` -> ""
    -   `None_value`: `None` -> ""

### `test_clean_text_with_stopwords`
-   **Goal**: Test text cleaning when stopword removal is enabled.
-   **Parametrized Cases**:
    -   `remove_stopwords`: Filters out "this", "is", "a" from "This is a test sentence".
    -   `remove_short_tokens`: Filters out words with <= 2 chars ("go", "to") when stopwords are provided.

---

## 3. Model Pipeline Tests (`test_model_pipeline.py`)

These tests verify the MLOps training logic without actually training a model (which would be slow and data-dependent).

### `test_train_baseline`
-   **Goal**: Validate the end-to-end execution of `train_baseline`.
-   **Method**: Uses `unittest.mock` to replace heavy external dependencies:
    -   `load_feature_data`: Returns dummy numpy arrays instead of reading files.
    -   `mlflow`: Captures calls to `start_run` and `log_model` without hitting a server.
    -   `save_model_bundle`: Verifies file saving logic without writing to disk.
-   **Assertions**:
    -   Check if data loader was called.
    -   Check if MLflow started a run.
    -   Check if `sklearn.log_model` and `log_metric` were invoked.
    -   Check if artifacts were "saved" (mock called).

---

## 4. Fixtures (`conftest.py`)

### `mock_params_yaml`
-   Creates a temporary YAML file mirroring the structure of `params.yaml`. This ensures tests don't rely on the actual project configuration, which might change.

### `config_manager`
-   Initializes a `ConfigurationManager` instance pointing to the temporary YAML. It also handles Singleton reset logic to prevent state leakage between tests.

---

## How to Run Tests

Run the full suite using `uv`:

```bash
uv run pytest
```

To run with verbose output:

```bash
uv run pytest -v
```
