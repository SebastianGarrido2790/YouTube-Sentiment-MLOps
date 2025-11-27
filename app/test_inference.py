"""
Automated Endpoint Test for YouTube Sentiment Analysis API
==========================================================

This test script validates that the unified FastAPI server (app/main.py)
is running correctly and returns valid JSON responses for sample inputs
across both general sentiment prediction and aspect-based sentiment analysis (ABSA) endpoints.

Usage:
Ensure FastAPI server is running:
    uv run uvicorn app.main:app --reload
In a new terminal, run the test client:
    uv run python -m app.test_inference

Responses:
1.  Sentiment Prediction:
    {
      "predictions": ["Positive"],
      "encoded_labels": [2],
      "feature_shape": [1, 1004]
    }

2.  Aspect-Based Sentiment:
    [
    {
        "aspect": "video quality",
        "sentiment": "positive",
        "score": 0.99...
    },
    {
        "aspect": "presenter",
        "sentiment": "negative",
        "score": 0.97...
    }
    ]
"""

import requests
import json
import sys

# Base URL for the FastAPI service
BASE_API_URL = "http://127.0.0.1:8000"

# Sample payloads for general sentiment prediction
PREDICT_TEST_PAYLOADS = [
    {
        "texts": ["I love this video! It was super helpful and well explained."]
    },  # Positive
    {"texts": ["This is terrible, I absolutely hate it."]},  # Negative
    {"texts": ["It’s okay, nothing special but not bad either."]},  # Neutral
]

# Sample payloads for Aspect-Based Sentiment Analysis (ABSA)
ABSA_TEST_PAYLOADS = [
    {
        "text": "The video quality was amazing, but the presenter seemed bored.",
        "aspects": ["video quality", "presenter"],
    },
    {
        "text": "The food was delicious, but the service was slow.",
        "aspects": ["food", "service"],
    },
]


def run_sentiment_test():
    """Sends test requests to the /predict endpoint."""
    print("\n🚀 Running general sentiment prediction API tests...\n")
    api_url = f"{BASE_API_URL}/predict"

    for payload in PREDICT_TEST_PAYLOADS:
        try:
            response = requests.post(api_url, json=payload, timeout=10)
            print(f"Input: {payload['texts'][0]}")

            if response.status_code == 200:
                result = response.json()
                print(json.dumps(result, indent=4))
            else:
                print(
                    f"❌ Request failed | Status {response.status_code} | {response.text}"
                )

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Connection error to {api_url}: {e}")
            sys.exit(1)

        print("-" * 80)


def run_absa_test():
    """Sends test requests to the /predict_absa endpoint."""
    print("\n🚀 Running Aspect-Based Sentiment Analysis (ABSA) API tests...\n")
    api_url = f"{BASE_API_URL}/predict_absa"

    for payload in ABSA_TEST_PAYLOADS:
        try:
            response = requests.post(api_url, json=payload, timeout=10)
            print(f"Input Text: {payload['text']}")
            print(f"Aspects: {payload['aspects']}")

            if response.status_code == 200:
                result = response.json()
                print(json.dumps(result, indent=4))
            else:
                print(
                    f"❌ Request failed | Status {response.status_code} | {response.text}"
                )

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Connection error to {api_url}: {e}")
            sys.exit(1)

        print("-" * 80)


if __name__ == "__main__":
    print("🧪 Starting FastAPI inference endpoint validation...\n")
    run_sentiment_test()
    run_absa_test()
    print("\n✅ All inference tests completed.\n")
