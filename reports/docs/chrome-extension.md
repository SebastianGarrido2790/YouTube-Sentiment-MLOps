# Chrome Extension for YouTube Sentiment Analysis

## 1. Overview

The Chrome extension is the user-facing component of the YouTube Sentiment Analysis project. It provides a simple interface to analyze the sentiment of comments on any YouTube video in real-time. The extension extracts comments from the video page, sends them to our FastAPI backend for analysis, and then displays a summary of the sentiments (Positive, Neutral, Negative) along with a pie chart visualization.

This document provides a guide for setting up, running, and developing the Chrome extension.

## 2. Core Components and Architecture

The extension is built with vanilla JavaScript, HTML, and CSS to keep it lightweight and secure. It consists of the following core files located in the `chrome-extension/` directory:

-   `manifest.json`: The extension's manifest file (V3), which defines its permissions and components.
-   `popup.html`: The HTML structure for the extension's popup UI.
-   `popup.css`: The styles for the popup UI.
-   `popup.js`: The main logic for the UI, including fetching comments and communicating with the backend.
-   `content_script.js`: A script that is injected into the YouTube video page to extract comments from the DOM.
-   `background.js`: A minimal service worker for future background tasks (optional for the current functionality).

### Data Flow

The extension's architecture follows a simple data flow:

1.  The user clicks the "Analyze Comments" button in the extension popup.
2.  `popup.js` sends a message to `content_script.js`.
3.  `content_script.js` scrapes the comments from the current YouTube page and sends them back to `popup.js`.
4.  `popup.js` sends the collected comments to the FastAPI `/predict` endpoint.
5.  The backend service processes the comments and returns sentiment predictions.
6.  `popup.js` receives the predictions and updates the UI to display the sentiment breakdown and pie chart.

## 3. Local Development Setup

Follow these steps to set up and run the Chrome extension on your local machine for development.

### Prerequisites

The backend services (FastAPI and MLflow) must be running. You can start them using Docker Compose:

```bash
# Run this command from the project root directory
docker compose -f docker/docker-compose.yml up --build -d
```

### Installation Steps

1.  **Configure CORS in the Backend**

    For the extension to be able to communicate with your local FastAPI server, you need to enable Cross-Origin Resource Sharing (CORS). Add the following middleware to `app/predict_model.py`:

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

2.  **Add Placeholder Icons**

    Chrome requires the icon files referenced in `manifest.json` to exist. Create an `icons` folder inside `chrome-extension/` and add placeholder PNG images (e.g., `icon16.png`, `icon48.png`, `icon128.png`).

3.  **Load the Extension in Chrome**
    -   Open Google Chrome and navigate to `chrome://extensions`.
    -   Enable "Developer mode" using the toggle in the top-right corner.
    -   Click the "Load unpacked" button.
    -   Select the `chrome-extension` folder from this project.

The extension should now be installed and visible in your Chrome toolbar.

## 4. How to Use

1.  Navigate to any YouTube video page with comments.
2.  Click on the YouTube Sentiment Analysis extension icon in the Chrome toolbar.
3.  Click the "Analyze Comments" button.
4.  The extension will display the sentiment analysis results in the popup.

## 5. Development and Configuration

### Backend Endpoint

The extension is configured to communicate with a backend running at `http://127.0.0.1:8000`. If you deploy the API to a different URL, you will need to update the `BACKEND_URL` constant in `popup.js` and the `host_permissions` in `manifest.json`.

### API Contract

The extension expects the `/predict` endpoint to accept a JSON payload with a `texts` field containing a list of comment strings:

```json
{
  "texts": ["This is a great video!", "I did not like this."]
}
```

The API should return a JSON object containing a `predictions` field with a list of sentiment labels.

### Data Source (DOM Scraping vs. YouTube Data API)

The current version of the extension **scrapes comments directly from the page's DOM**. This method is simple and does not require an API key.

However, for a more robust and compliant solution, future development should consider using the official **YouTube Data API**. This would require obtaining an API key and managing quotas but would provide more reliable access to comments and other video metadata. The `youtube_api.js` file contains a sample implementation for this approach.
