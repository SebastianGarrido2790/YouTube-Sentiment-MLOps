# Technical Implementation Report: Data Validation

**Stage:** 01b — Data Validation  
**Focus:** Implementation Details ("The How")  
**Toolkit:** Great Expectations (GX) + Pandas

---

## 1. Overview
The Data Validation implementation ensures that the raw data artifacts conform to the statistical and structural definitions defined in the **Data Contract** (`schema.yaml`). It utilizes an ephemeral configuration strategy to integrate the heavy-weight **Great Expectations** SDK into a lightweight, modular pipeline stage.

---

## 2. Technical Stack
- **Framework:** `great_expectations` (GX) v1.0+.
- **Data Engine:** `Pandas`.
- **Validation Logic:** Ephemeral in-memory context.
- **Persistence:** JSON serialization to local artifacts.

---

## 3. Implementation Workflow

### 3.1 Ephemeral Context Initialization
To avoid the complexity of managing a `gx/` directory with static YAML files, the component uses the **Ephemeral Context** pattern. This allows the system to be stateless and portable.

```python
import great_expectations as gx
context = gx.get_context() # Initializes an in-memory ephemeral context
```

### 3.2 Data Asset Mapping
The local raw CSV file is wrapped as a Pandas data asset within the GX context:
1. **Data Source:** `context.data_sources.add_pandas("youtube_sentiment_source")`
2. **Data Asset:** `data_source.add_dataframe_asset("raw_youtube_comments")`
3. **Batch Definition:** `data_asset.add_batch_definition_whole_dataframe("raw_batch")`

### 3.3 Dynamic Suite Generation
The `DataValidation` component dynamically builds its `ExpectationSuite` by iterating over the schema and parameters.

**A. Structural Mapping (from schema.yaml):**
```python
expected_columns = list(self.schema.columns.keys())
for col in expected_columns:
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))
```

**B. Quality Mapping (from params.yaml):**
The `null_threshold_percent` is converted into a GX `mostly` parameter (the fraction of data that must pass):
```python
mostly = 1.0 - (self.config.null_threshold_percent / 100.0)
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(column="clean_comment", mostly=mostly)
)
```

---

## 4. Manual Persistence Logic
Because the GX context is ephemeral, it does not automatically save suites or results to disk. The implementation manually bridges this gap using standard JSON serialization:

```python
# Save Expectation Suite
contract_path = os.path.join(self.ops_paths.gx_dir, f"{suite_name}.json")
with open(contract_path, "w") as f:
    json.dump(suite.to_json_dict(), f, indent=4)

# Save Validation Results
result_path = os.path.join(self.ops_paths.gx_dir, "validation_results.json")
with open(result_path, "w") as f:
    json.dump(validation_results.to_json_dict(), f, indent=4)
```

---

## 5. Error Handling & Robustness

### 5.1 Defensive Checks
Before invoking GX (which is resource-intensive), the component performs two defensive checks:
- **File Existence:** Verifies the raw CSV exists at the expected path.
- **Content Check:** Uses `pd.read_csv` inside a `try-except` to catch `EmptyDataError`.

### 5.2 Exception Management
The Conductor (`stage_01b_...py`) catches any uncaught exceptions from the GX engine and logs them via `logger.exception(e)`. This ensures that even "engine-level" failures (like OOM or library conflicts) are logged with a full traceback and surfaced to the orchestrator.

---

## 6. Integration Points

### 6.1 Parameter Injection
The thresholds are injected via the `ConfigurationManager`:
- `val_config.null_threshold_percent`
- `val_config.min_text_length`
- `val_config.max_text_length`

### 6.2 DVC Outputs
The implementation targets the `artifacts/gx/` directory. DVC is configured to track this entire directory, ensuring that for every dataset version, there is a matching validation evidence file.

---

## 7. Execution Logic Summary
1.  **Prepare:** Initialize `ConfigurationManager` and fetch merged configs.
2.  **Inspect:** Load raw CSV into a DataFrame and perform a "Fast Fail" empty check.
3.  **Context:** Start an ephemeral GX context and define the Pandas data route.
4.  **Contract:** Build an `ExpectationSuite` from schema and quality params.
5.  **Audit:** Manually serialize the suite JSON for documentation.
6.  **Execute:** Run the `ValidationDefinition` against the batch.
7.  **Evidence:** Serialize the full result JSON and return the `success` boolean.
