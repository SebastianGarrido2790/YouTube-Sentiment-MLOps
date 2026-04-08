# Chrome Extensions for YouTube Sentiment Analysis

## 1. Overview

This project includes two user-facing Chrome extensions for real-time sentiment analysis of YouTube comments. Each extension provides a simple interface to analyze comments, but they target different levels of detail:

1.  **Standard Sentiment Analysis Extension:** Provides a high-level summary of sentiments (Positive, Neutral, Negative) for all comments.
2.  **Aspect-Based Sentiment Analysis (ABSA) Extension:** Offers a more granular analysis by identifying sentiment towards specific topics or "aspects" (e.g., "video quality," "presenter") defined by the user.

This document provides a guide for setting up, running, and developing both extensions.

## 2. API Key Management

The location to save your API key depends on which part of the system you are targeting. Because the Chrome extension and the MLOps backend are independent, you now manage the key in two different places:

### **1. Chrome Extension (Popup UI)**
Since my last update, **you no longer need to edit the source code.** 
*   **Where to save:** Open the **YouTube Sentiment Insights** extension on any YouTube page. At the top, you'll see a **YouTube API Key** field under the *Settings* section. 
*   **How:** Paste your key there and click **Save**. The extension will store this securely in your browser's local storage (`chrome.storage.local`).

### **2. MLOps Backend & Pipelines (`.env` file)**
For the Python-based data ingestion and training pipelines (the "Hands" that execute the FTI pattern), the key is still managed via an environment file.
*   **Where to save:** In the project root, create or edit the `.env` file (copied from [`.env.example`](../../../.env.example)).
*   **The Line:**
    ```env
    YOUTUBE_API_KEY=your_real_api_key_here
    ```
*   **Why:** This ensures that when you run `dvc repro` or initiate a training stage, the backend can fetch historical comments and perform quantitative lifting without human intervention.

> [!TIP]
> This "Dual-Key" approach follows the **Separation of Concerns** principle: the extension handles user-facing real-time analysis (Probabilistic Brain), while the backend handles persistent MLOps data ingestion (Deterministic Brawn).

## 3. Available Extensions

### Standard Sentiment Analysis Extension (`chrome-extension/`)

This extension gives a general overview of the sentiment expressed in the comments section. It's ideal for quickly gauging the overall community reaction to a video.

-   **Functionality:** Extracts comments, sends them to the `/predict` endpoint, and visualizes the aggregated results (Positive, Neutral, Negative) in a pie chart.
-   **Location:** `chrome-extension/`

### Aspect-Based Sentiment Analysis (ABSA) Extension (`chrome-extension-absa/`)

This advanced extension allows users to analyze sentiment with respect to specific features or topics mentioned in the comments. For example, you can check what users think about the "audio quality" or the "host" of a podcast.

-   **Functionality:** Extracts comments, combines them into a single text block, and sends it to the `/predict_absa` endpoint along with a list of user-defined aspects. It then displays the sentiment for each aspect.
-   **Location:** `chrome-extension-absa/`

## 4. Core Components and Architecture

Both extensions are built with vanilla JavaScript, HTML, and CSS to keep them lightweight and secure. They share an identical file structure:

-   `manifest.json`: The extension's manifest file (V3), defining its permissions and components.
-   `popup.html`: The HTML structure for the extension's popup UI.
-   `popup.css`: The styles for the popup UI.
-   `popup.js`: The main logic for the UI, including fetching comments and communicating with the backend.
-   `youtube_api.js`: Contains a sample implementation for using the official YouTube Data API (currently not the primary method).

### Data Flow

1.  The user clicks the "Analyze" button in the extension popup.
2.  `popup.js` uses Chrome's scripting API to execute a function that scrapes comments from the page.
3.  The scraped comments are sent to the appropriate FastAPI endpoint (`/predict` or `/predict_absa`).
4.  The backend service returns the sentiment predictions.
5.  `popup.js` receives the predictions and updates the UI to display the results.

## 5. Local Development Setup

Follow these steps to set up and run the extensions on your local machine.

### Prerequisites

The backend services must be running. You can start them using the appropriate prediction script (e.g., `predict_model_absa.py` which serves both endpoints):

```bash
# Ensure you are in the project root directory
uv run python -m app.predict_model_absa
```

### Installation Steps

1.  **Configure CORS in the Backend**

    For the extensions to communicate with your local server, you must enable Cross-Origin Resource Sharing (CORS) in your FastAPI application. Add the middleware to `app/predict_model_absa.py`:

    ```python
    from fastapi.middleware.cors import CORSMiddleware

    # Add after creating the app instance
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For development only. Restrict in production.
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    ```

2.  **Load the Extensions in Chrome**
    -   Open Google Chrome and navigate to `chrome://extensions`.
    -   Enable "Developer mode" using the toggle in the top-right corner.
    -   Click the "Load unpacked" button.
    -   **To load the standard extension:** Select the `chrome-extension` folder.
    -   **To load the ABSA extension:** Click "Load unpacked" again and select the `chrome-extension-absa` folder.

You should now see both extensions in your Chrome toolbar.

## 6. How to Use the Extensions

### Standard Extension

1.  Navigate to any YouTube video page with comments.
2.  Click on the standard extension icon.
3.  Click the "Analyze Comments" button.
4.  View the aggregated sentiment results in the pie chart.

### ABSA Extension

1.  Navigate to any YouTube video page.
2.  Click on the ABSA extension icon.
3.  In the input field, enter the aspects you want to analyze, separated by commas (e.g., `video quality, presenter, audio`).
4.  Click the "Analyze Aspects" button.
5.  View the sentiment breakdown for each aspect.

## 7. Development and Configuration

### Backend Endpoint

The extensions are configured to communicate with a backend at `http://127.0.0.1:8000`. This URL is defined in the respective `popup.js` files.

-   **Standard Extension:** Communicates with the `/predict` endpoint.
-   **ABSA Extension:** Communicates with the `/predict_absa` endpoint.

### API Contract

-   The `/predict` endpoint expects `{"texts": ["comment1", "comment2"]}`.
-   The `/predict_absa` endpoint expects `{"text": "full comments text", "aspects": ["aspect1", "aspect2"]}`.

### Data Source (DOM Scraping vs. YouTube Data API)

Both extensions currently **scrape comments directly from the page's DOM**. This method is simple and avoids the need for an API key.

For a more robust and compliant solution, future development could use the official **YouTube Data API**. This would provide more reliable access to comments. The `youtube_api.js` file in each extension directory contains a sample implementation for this approach.
