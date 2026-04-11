# Technical Implementation Report: DistilBERT Fine-Tuning Pipeline

**Stage:** 09 — DistilBERT Training  
**Focus:** Implementation Details ("The How")  
**Toolkit:** PyTorch + Transformers + Optuna + MLflow

---

## 1. Overview
The DistilBERT implementation provides a high-fidelity deep learning path for sentiment analysis. It solves the "Computation Wall" (high VRAM requirements) through adaptive training strategies and ensures pipeline continuity in CPU-only environments through an automated fallback mechanism.

---

## 2. Technical Stack
- **Framework:** `torch` (CUDA 12.1+ recommended).
- **Model Architecture:** `distilbert-base-uncased`.
- **Optimization:** `HuggingFace Trainer` + `Optuna`.
- **Dataset Handling:** `CustomDataset` (mapped from Pandas).

---

## 3. Implementation Workflow

### 3.1 Adaptive Execution Gate
The implementation begins with a strict boolean verification:
- **CUDA Check:** `torch.cuda.is_available()`.
- **Config Check:** Retrieves the `enable` flag from `params.yaml`.
If either is false, the implementation triggers `handle_placeholder_stage()`, which creates a valid directory structure with null artifacts to satisfy DVC `outs` without performing expensive compute.

### 3.2 Feature Tokenization
The text is processed via a `DistilBertTokenizer`:
- **Padding:** `max_length` (512 tokens).
- **Truncation:** `True` (to handle long YouTube comments).
- **Torch Dataset:** Implements a `YoutubeSentimentDataset` wrapper that lazily provides inputs (`input_ids`, `attention_mask`) and labels to the Trainer.

### 3.3 Fine-Tuning Logic (Active Path)
The active training path implements a 3rd-party optimization loop:
- **Mixed Precision:** Uses `fp16=True` to accelerate training and reduce VRAM footprint by 40-50%.
- **Gradient Accumulation:** Dynamically adjusts effective batch size to maintain stable gradients on low-memory GPUs.
- **Optuna Objective:** The `hp_space` suggest `learning_rate` between `2e-5` and `5e-5` to prevent catastrophic forgetting of the pre-trained weights.

---

## 4. Telemetry & Packaging

### 4.1 MLflow Integration
Each trial is logged as a nested run. The implementation specifically filters for the **Global Step** to avoid redundant logging of intermediate epoch metrics, keeping the MLflow database light.

### 4.2 The "Null" Model Bundle
In placeholder mode, the system pickles a dictionary `{"model": None, "encoder": le}`. This ensures that the downstream `model_evaluation` component can detect the absence of the model (at the object level) and skip the DistilBERT column in the performance comparison table gracefully.

---

## 5. Robustness & Optimization

- **VRAM Guard:** The implementation uses `per_device_train_batch_size: 8` as a safe default, with automated gradient accumulation to simulate a batch of 32 or 64.
- **Early Stopping:** Monitors `eval_loss` with a patience of 3 epochs to prevent overfitting on small labels.
- **Weight Decay:** Fixed at `0.01` to provide L2-style regularization for the transformer's multi-head attention weights.

---

## 6. Execution Logic Summary
1.  **Gatekeeper:** Verify hardware and configuration flags.
2.  **Ingest (Active):** Load raw text splits and tokenize using the DistilBERT vocabulary.
3.  **Optimize (Active):** Run Optuna trials to find the ideal learning rate.
4.  **Train (Active):** Execute the final Fine-Tuning sweep using the HuggingFace Trainer API.
5.  **Fallback (Passive):** Generate placeholder objects if prerequisites are not met.
6.  **Bundle:** Save the estimator state and label encoder to `artifacts/distilbert_model/`.
7.  **Log:** Stream telemetry to MLflow for cross-model benchmarking.
