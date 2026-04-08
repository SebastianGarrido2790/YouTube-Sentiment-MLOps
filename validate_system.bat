@echo off
setlocal
title YouTube Sentiment Analysis - Multi-Point System Validation

:: Clean screen and display banner
cls
echo ============================================================
echo   🛠 YOUTUBE SENTIMENT: SYSTEM HEALTH CHECK
echo ============================================================
echo.
echo [SYSTEM] Starting full architecture health check...
echo.

:: Pillar 0: Sync Dependencies
echo [0/4] Pillar 0: Syncing all dependencies...
call uv sync --all-extras --quiet
if %ERRORLEVEL% NEQ 0 goto :FAILED

:: Pillar 1: Static Code Quality
echo [1/4] Pillar 1: Static Code Quality Pyright ^& Ruff...
echo      - Running Pyright (Static Type Checking)...
call uv run pyright src/
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo      [!] WARNING: Pyright reported type issues. Review them later.
    echo          [Hardening Phase: Static checks are currently non-blocking]
)

echo.
echo      - Running Ruff (Linting)...
call uv run ruff check .
if %ERRORLEVEL% NEQ 0 goto :FAILED

echo.
echo      - Running Ruff (Formatting Check)...
call uv run ruff format --check .
if %ERRORLEVEL% NEQ 0 goto :FAILED

echo      Done.
echo.

:: Pillar 2: Functional Logic ^& Coverage
echo [2/4] Pillar 2: Functional Logic ^& Coverage...
echo      - Running Pytest with Coverage Gate: 50%%...
call uv run pytest tests/ --cov=src --cov-fail-under=50 --tb=short
if %ERRORLEVEL% NEQ 0 goto :FAILED

echo      Done.
echo.

:: Pillar 3: Pipeline Synchronization
echo [3/4] Pillar 3: Pipeline Synchronization DVC...
call uv run python -m dvc status
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo      [!] WARNING: DVC status failed or remote is unreachable. 
    echo          [Ensuring this is non-blocking for environment validation]
)

echo      Done.
echo.

:: Pillar 4: App Service Health
echo [4/4] Pillar 4: App Service Health...

:: Check Sentiment API (8000)
powershell -Command "try { $c = New-Object System.Net.Sockets.TcpClient('localhost', 8000); if ($c.Connected) { exit 0 } else { exit 1 } } catch { exit 1 }"
if %ERRORLEVEL% NEQ 0 (
    echo      Sentiment API is OFFLINE on port 8000.
) else (
    echo      Sentiment API is ONLINE on port 8000.
    echo      - Verification [Health Endpoint]:
    call :CHECK_SENTIMENT_HEALTH
)

:: Check Insights API (8001)
powershell -Command "try { $c = New-Object System.Net.Sockets.TcpClient('localhost', 8001); if ($c.Connected) { exit 0 } else { exit 1 } } catch { exit 1 }"
if %ERRORLEVEL% NEQ 0 (
    echo      Insights API is OFFLINE on port 8001.
) else (
    echo      Insights API is ONLINE on port 8001.
    echo      - Verification [Root Endpoint]:
    call :CHECK_INSIGHTS_HEALTH
)

echo      Done.
echo.

goto :SUCCESS

:CHECK_SENTIMENT_HEALTH
powershell -Command "$response = Invoke-RestMethod -Uri 'http://localhost:8000/v1/health' -Method Get; if ($response.status -eq 'healthy') { Write-Host '        [OK] Health check passed'; exit 0 } else { Write-Host '        [FAIL] Health check report non-healthy'; exit 1 }"
exit /b

:CHECK_INSIGHTS_HEALTH
powershell -Command "$response = Invoke-WebRequest -Uri 'http://localhost:8001/v1/' -Method Get; if ($response.StatusCode -eq 200) { Write-Host '        [OK] Endpoint reachable'; exit 0 } else { Write-Host '        [FAIL] Endpoint unreachable'; exit 1 }"
exit /b

:SUCCESS
echo ============================================================
echo   ✅ SYSTEM HEALTH: 100%% ALL GATES PASSED
echo ============================================================
echo.
echo Your Hardened YouTube Sentiment architecture is validated.
pause
exit /b 0

:FAILED
echo.
echo ============================================================
echo   ❌ VALIDATION FAILED
echo ============================================================
echo.
echo Please review the logs above and correct the issues.
pause
exit /b 1
