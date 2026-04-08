# Chrome Extensions for YouTube Sentiment Analysis

## 1. Overview

This project includes two user-facing Chrome extensions for real-time sentiment analysis of YouTube comments. These extensions leverage specialized backend APIs to transform raw comment data into actionable insights:

1.  **Standard Insights Extension:** Provides high-level analytics, including sentiment distribution (pie charts), monthly trends (line graphs), and word clouds.
2.  **Aspect-Based Sentiment Analysis (ABSA) Extension:** Offers granular analysis by identifying sentiment towards specific user-defined topics (e.g., "video quality").

This document provides a guide for setting up, running, and developing both extensions.

## 2. API Key Management

The location to save your API key depends on the component. The Chrome extensions and the MLOps backend maintain independent contexts for flexibility.

### **1. Chrome Extension (Popup UI)**
*   **Where to save:** Open the extension popup on any YouTube page. You'll find a **YouTube API Key** field under the *Settings* section.
*   **How:** Paste your key and click **Save**. The extension stores this securely via `chrome.storage.local`.
*   **Usage:** This key is used by `youtube_api.js` to fetch comments directly from the YouTube Data API V3.

### **2. MLOps Backend Pipelines (`.env` file)**
*   **Where to save:** In the project root, edit the `.env` file (see `.env.example`).
*   **The Line:** `YOUTUBE_API_KEY=your_real_api_key_here`
*   **Why:** This allows the data ingestion pipelines (e.g., `stage_01_data_ingestion`) to fetch large datasets for model training and validation.

## 3. Extension Architecture

### Standard Insights Extension (`chrome-extension/`)
Dedicated to dashboard-style visualization and trend analysis.
-   **Backend Port:** `8001` (Insights API)
-   **Core Interface:** `popup.js` ↔ `src.api.insights_api`
-   **Key Endpoints:**
    -   `/v1/predict_with_timestamps`: Fetches raw sentiment scores.
    -   `/v1/generate_chart`: Returns a dynamically generated pie chart.
    -   `/v1/generate_wordcloud`: Returns a term-frequency word cloud image.
    -   `/v1/generate_trend_graph`: Returns a monthly sentiment trend line.

### ABSA Extension (`chrome-extension-absa/`)
Focused on granular, topic-specific qualitative analysis.
-   **Backend Port:** `8000` (Main API)
-   **Core Interface:** `popup.js` ↔ `src.api.main`
-   **Key Endpoints:**
    -   `/v1/predict_absa`: Evaluates specific aspects within comment text.

## 4. Quick Start: The Unified Launcher

For local development and testing, a unified batch script is provided to initialize all required backend services with a single command. This ensures all ports (5000, 8000, and 8001) are correctly managed and synchronized.

*   **File:** [`launch_extension_backend.bat`](../../../launch_extension_backend.bat)
*   **What it does:**
    1.  Syncs dependencies via `uv sync`.
    2.  Starts the **MLflow Server** (Port 5000) minimized.
    3.  Starts the **Main Inference API** (Port 8000) minimized.
    4.  Starts the **Insights Visualization API** (Port 8001) in the foreground for real-time log monitoring.

**To run:** Simply double-click the file in the project root or run `.\launch_extension_backend.bat` from your terminal.

## 5. Local Development Setup (Manual)

### Prerequisites
While the **Unified Launcher** is the recommended method, you can also start services individually if you need to debug specific components:

```bash
# Terminal 1: MLflow Tracking Server (Port 5000)
uv run python -m mlflow server --backend-store-uri sqlite:///mlflow_system.db --default-artifact-root ./mlruns_system --host 127.0.0.1 --port 5000

# Terminal 2: Main Inference API (Port 8000)
uv run python -m src.api.main

# Terminal 3: Insights Visualization API (Port 8001)
uv run python -m src.api.insights_api
```

### Installation Steps
1.  **Enable Developer Mode**: Open Chrome and navigate to `chrome://extensions`. Toggle **Developer mode** on.
2.  **Load Extensions**: Click **Load unpacked** and select the extension folders:
    -   Select `chrome-extension/` for the Insights tool.
    -   Select `chrome-extension-absa/` for the ABSA tool.
3.  **Refetch Updates**: If you make changes to the extension's JavaScript or the documentation, remember to click the **Refresh (⟳)** icon in `chrome://extensions`.

## 6. File Structure
Both extensions follow a clean, modular structure:
-   `manifest.json`: Extension metadata and permissions (V3).
-   `popup.html/css`: Premium UI designed for readability.
-   `popup.js`: Main orchestration logic; manages state and API communication.
-   `youtube_api.js`: Handles authenticated requests to the YouTube Data API.

## 6. API Contracts

### Insights API (v1)
-   **Input**: `{"comments": [{"text": "string", "timestamp": "ISO_timestamp"}]}`
-   **Output**: JSON predictions or binary PNG images (for graphs).

### Main API (v1)
-   **Input**: `{"text": "string", "aspects": ["list", "of", "strings"]}`
-   **Output**: List of aspect-sentiment objects `{"aspect": "string", "sentiment": "string", "score": float}`.

## 7. Troubleshooting
If the extension shows a "Connection Error":
1.  Verify the backend servers are running (check ports 8000 and 8001).
2.  Ensure your **YouTube API Key** is saved in the extension settings.
3.  Check the browser console (Right-click Extension Popup -> Inspect) for specific HTTP error codes (e.g., 404 implies a missing prefix, 422 implies a schema mismatch).
