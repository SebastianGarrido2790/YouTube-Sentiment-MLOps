# Technical Implementation Report: Data Ingestion

**Stage:** 01 — Data Ingestion  
**Focus:** Implementation Details ("The How")  
**Architecture Pattern:** Conductor-Worker Separation

---

## 1. Overview
The Data Ingestion phase is implemented as a decoupled system consisting of a **Pipeline Conductor** that handles orchestration and a **Worker Component** that executes the specialized task of data retrieval. This design ensures that the business logic (downloading) is completely independent of the orchestration tool (DVC, FastAPI, or CLI).

---

## 2. Technical Stack
- **HTTP Client:** `requests` (with streaming enabled).
- **Configuration:** `Pydantic` (Data Contracts) + `PyYAML`.
- **Orchestration:** `FastAPI` (Async) + `DVC` (DAG).
- **Environment:** `Python 3.12` / `uv`.

---

## 3. Class Structure & Design

### 3.1 DataIngestionConfig (`src/entity/config_entity.py`)
Provides strict schema enforcement using Pydantic's `BaseModel`.

```python
class DataIngestionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    url: str         # Remote source
    output_path: str # Local destination
```

### 3.2 DataIngestion Component (`src/components/data_ingestion.py`)
The "worker" class that implements the low-level logic.

**Key Implementation Features:**
- **Directory Auto-Creation:** Uses `os.makedirs(output_dir, exist_ok=True)` to ensure the landing zone exists before downloading.
- **Memory-Efficient Streaming:** 
  Instead of loading the entire dataset into RAM, it uses a chunked write strategy:
  ```python
  response = requests.get(url, stream=True, timeout=30)
  with open(output_path, "wb") as f:
      for chunk in response.iter_content(chunk_size=8192):
          f.write(chunk)
  ```
  - `stream=True`: Defers downloading the response body until it's accessed.
  - `chunk_size=8192`: Writes 8KB at a time, protecting the system from large file memory overflows.

### 3.3 DataIngestionPipeline Conductor (`src/pipeline/stage_01_data_ingestion.py`)
The "conductor" class that bridges configuration and execution.

**Responsibilities:**
- Initializes the `ConfigurationManager` (Singleton).
- Gracefully handles missing configs with logged fallbacks (for local debugging).
- Instantiates the `DataIngestion` component with the valid `DataIngestionConfig` object.

---

## 4. Configuration Pipeline
The system avoids hardcoded strings through a centralized configuration manager.

1.  **Loader:** `ConfigurationManager` reads `params.yaml`.
2.  **Parser:** The YAML dictionary is passed to `AppConfig` (root schema).
3.  **Validation:** Pydantic validates the URL format and types.
4.  **Injection:** The specific `DataIngestionConfig` sub-object is injected into the component.

---

## 5. Error Handling & Robustness

### 5.1 Exception Bubbling
The component catches `requests.exceptions.RequestException` to log a detailed error message but re-raises the exception.
- This allows the **FastAPI Orchestrator** (`main.py`) to detect the failure.
- Triggering the orchestrator's **Retry Logic** (exponential backoff).

### 5.2 logging Strategy
Uses a custom header-based logger (`src.utils.logger`) to visually separate stages in the logs:
```
[2026-04-11 14:00:00] [INFO] [data_ingestion_component] --------------------
[2026-04-11 14:00:00] [INFO] [data_ingestion_component] Loading configuration...
[2026-04-11 14:00:01] [INFO] [data_ingestion_component] Downloading data from: https://...
[2026-04-11 14:00:02] [INFO] [data_ingestion_component] ✅ Successfully saved to data/raw/
```

---

## 6. Integration with MLOps Tools

### 6.1 DVC Integration
The entry point `__main__` in the conductor script allows DVC to execute the stage as a module:
`python -m src.pipeline.stage_01_data_ingestion`

### 6.2 AgentOps Metrics
The `main.py` orchestrator measures the implementation's success through the following metrics snapshot:
- **Total Success:** If `pipeline.main()` finishes without an exception.
- **Failures:** Logged into `AGENT_METRICS.failed_tool_calls` for auditing.
- **Validation:** Output existence is verified before proceeding to Stage 01b (Validation).

---

## 7. Execution Logic Summary
1.  **Initialize:** Load `ConfigurationManager`.
2.  **Plan:** Retrieve `DataIngestionConfig`.
3.  **Prepare:** Create `data/raw/` directory.
4.  **Execute:** Stream HTTPS GET request to file system in 8KB chunks.
5.  **Verify:** `raise_for_status()` check and log completion.
