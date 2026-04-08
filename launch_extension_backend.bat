@echo off
setlocal
title YouTube Sentiment Analysis - Launcher

:: Clean screen and display banner
cls
echo ============================================================
echo   🎥 YOUTUBE SENTIMENT ANALYSIS: AGENTIC MLOPS SYSTEM
echo ============================================================
echo.
echo [SYSTEM] Initializing Backend Services for Chrome Extensions...
echo.

:: Step 1: Check/Sync Dependencies
echo [1/4] Verifying dependencies with UV...
uv sync --quiet
if "%ERRORLEVEL%" NEQ "0" (
    echo.
    echo 🚨 Error: Failed to sync dependencies. Verify 'uv' is installed.
    pause
    exit /b %ERRORLEVEL%
)
echo      Done.
echo.

:: Step 2: Launch MLflow Tracking Server
echo [2/4] Launching MLflow Tracking Server...
echo      URL: http://127.0.0.1:5000
echo      Storage: sqlite:///mlflow_system.db
start "YT-MLFLOW" /min cmd /k "title YT-MLFLOW && uv run python -m mlflow server --backend-store-uri sqlite:///mlflow_system.db --default-artifact-root ./mlruns_system --host 127.0.0.1 --port 5000"

:: Step 3: Launch Main Inference API (Port 8000)
echo [3/4] Launching Main API (ABSA Support)...
echo      URL: http://127.0.0.1:8000
start "YT-MAIN-API" /min cmd /k "title YT-MAIN-API && uv run python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload"

:: Wait for warmup
echo.
echo [WAIT] Stalling for service initialization (5s)...
timeout /t 5 >nul

:: Step 4: Launch Insights API in the foreground (Port 8001)
echo.
echo [4/4] Launching Insights API (Standard Dashboard)...
echo      URL: http://127.0.0.1:8001
echo.
echo ------------------------------------------------------------
echo 💡 TIP: The MLflow server and Main API are running in
echo    the background (minimized).
echo.
echo    To stop EVERYTHING:
echo    1. Close the "YT-MLFLOW" and "YT-MAIN-API" windows.
echo    2. Press Ctrl+C in this window.
echo ------------------------------------------------------------
echo.

:: Run Insights API in foreground so user sees primary logs
uv run python -m uvicorn src.api.insights_api:app --host 127.0.0.1 --port 8001 --reload

:: If the user stops the API
echo.
echo [SYSTEM] Backend Services Terminated.
pause
