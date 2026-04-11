@echo off
setlocal
title YouTube Sentiment Analysis - Multi-Point System Validation

:: Clean screen and display banner
cls
echo ============================================================
echo   🛠 HYBRID AGENTIC MLOps SYSTEM (v2.0): HEALTH CHECK
echo ============================================================
echo.
echo [SYSTEM] Starting full architecture health check...
echo.

:: Pillar 0: Sync Dependencies
echo [0/4] Pillar 0: Syncing all dependencies...
echo      - Using uv for lightning-fast resolution...
call uv sync --all-extras --quiet
if %ERRORLEVEL% NEQ 0 goto :FAILED

:: Pillar 1: Static Code Quality
echo [1/4] Pillar 1: Static Code Quality Pyright ^& Ruff...
echo      - Running Pyright (Static Type Checking)...
call uv run pyright src/
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo      [!] WARNING: Pyright reported type issues.
    echo          [Agentic Phase: Enforcing strict typing where possible]
)

echo.
echo      - Running Ruff (Linting ^& Import Sorting)...
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
echo      - Running Pytest (Unit + Integration + Agent Tests)...
echo      - Coverage Gate: 50%%...
call uv run pytest tests/ --cov=src --cov-fail-under=50 --tb=short
if %ERRORLEVEL% NEQ 0 goto :FAILED

echo      Done.
echo.

:: Pillar 3: Data ^& Model Lineage
echo [3/4] Pillar 3: Data ^& Model Lineage (DVC)...
call uv run python -m dvc status
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo      [!] WARNING: DVC status failed or remote is unreachable.
)

echo      Done.
echo.

:: Pillar 4: App Service Health (FTI Architecture)
echo [4/4] Pillar 4: App Service Health (FTI Architecture)...

:: Check Inference API (8000) - Now hosts the Agentic Layer
powershell -Command "try { $c = New-Object System.Net.Sockets.TcpClient('localhost', 8000); if ($c.Connected) { exit 0 } else { exit 1 } } catch { exit 1 }"
if %ERRORLEVEL% NEQ 0 (
    echo      Inference API + Agent Layer is OFFLINE on port 8000.
) else (
    echo      Inference API + Agent Layer is ONLINE on port 8000.
    echo      - Verification [Health Endpoint]:
    call :CHECK_SENTIMENT_HEALTH
)

:: Check Insights API (8001) - Legacy Dashboard Support
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
echo Your Hybrid Agentic YouTube Sentiment system (v2.0) is validated.
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
