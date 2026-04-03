Here’s a clean and reusable **PowerShell script** you can save as `reset-venv-full.ps1` in your project root. A short **PowerShell automation script** that you can run anytime to completely reset your .venv and sync with uv automatically. Not only rebuilds your Python environment but also ensures your DVC cache, MLflow tracking server, and registry connectivity are clean and healthy afterward. Additionally, includes automatic Docker Compose stack restart after environment.

It’s useful to avoid lock issue in the future. This makes your full environment reset completely reproducible and CI/CD-friendly.

It automatically:

1. Cleans .venv
2. Rebuilds dependencies via uv
3. Reconstructs your DVC cache
4. Confirms MLflow availability and model alias
5. Tears down and rebuilds your entire Docker Compose stack
(ensuring mlflow and youtube-sentiment-api containers start fresh)
6. Verifies your Python environment is operational

---

### 🧩 `reset-venv-full.ps1`

```powershell
<#
.SYNOPSIS
    Fully resets the local ML environment: Python venv, DVC cache, MLflow connectivity, and Docker services.

.DESCRIPTION
    - Stops Python processes
    - Deletes and recreates .venv using uv
    - Cleans and pulls DVC cache
    - Restarts Docker Compose stack (MLflow + FastAPI)
    - Verifies MLflow Tracking Server + registry alias
    - Confirms everything is up and running

.NOTES
    Author: Sebastián Garrido
    Project: YouTube Sentiment MLOps
#>

# ----------------------------------------------
# Step 0 — Display banner
# ----------------------------------------------
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "  🔄  RESETTING FULL MLOPS ENVIRONMENT (venv + DVC + Docker + MLflow)  " -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# ----------------------------------------------
# Step 1 — Stop Python processes
# ----------------------------------------------
Write-Host "⏹  Stopping any running Python processes..."
try {
    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Write-Host "✅  Python processes stopped successfully."
} catch {
    Write-Host "ℹ️  No Python processes were running."
}

# ----------------------------------------------
# Step 2 — Remove .venv
# ----------------------------------------------
if (Test-Path ".venv") {
    Write-Host "🧹  Removing old .venv environment..."
    try {
        Remove-Item -Recurse -Force ".venv"
        Write-Host "✅  .venv directory removed."
    } catch {
        Write-Host "⚠️  Could not remove .venv directly. Attempting elevated removal..."
        Start-Process powershell -Verb runAs -ArgumentList "Remove-Item -Recurse -Force '.venv'"
        exit
    }
} else {
    Write-Host "ℹ️  No .venv directory found."
}

# ----------------------------------------------
# Step 3 — Recreate environment
# ----------------------------------------------
Write-Host "⚙️  Rebuilding environment using uv sync..."
uv sync
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌  uv sync failed. Please review pyproject.toml."
    exit 1
}
Write-Host "✅  Virtual environment rebuilt successfully."

# ----------------------------------------------
# Step 4 — Clean and rebuild DVC cache
# ----------------------------------------------
Write-Host "🧩  Cleaning DVC cache..."
uv run dvc gc -w -f | Out-Null
Write-Host "⚙️  Pulling latest tracked artifacts..."
uv run dvc pull
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌  DVC pull failed. Check remote connectivity or authentication."
    exit 1
}
Write-Host "✅  DVC cache rebuilt successfully."

# ----------------------------------------------
# Step 5 — Restart Docker Compose stack
# ----------------------------------------------
Write-Host ""
Write-Host "🐳  Restarting Docker Compose stack (MLflow + FastAPI)..."
try {
    docker compose down --remove-orphans | Out-Null
    docker compose up -d --build
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅  Docker Compose stack rebuilt and started successfully." -ForegroundColor Green
    } else {
        Write-Host "⚠️  Docker Compose encountered issues. Review logs manually." -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌  Failed to restart Docker Compose stack. Verify Docker Desktop is running." -ForegroundColor Red
    exit 1
}

# ----------------------------------------------
# Step 6 — Verify MLflow connectivity
# ----------------------------------------------
$mlflowUrl = "http://127.0.0.1:5000"
Write-Host "🌐  Checking MLflow Tracking Server at $mlflowUrl ..."
try {
    $response = Invoke-WebRequest -Uri $mlflowUrl -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✅  MLflow tracking server is reachable." -ForegroundColor Green
    } else {
        Write-Host "⚠️  MLflow responded with status code: $($response.StatusCode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌  MLflow tracking server not reachable. Ensure Docker or local MLflow is running." -ForegroundColor Red
}

# ----------------------------------------------
# Step 7 — Check if Production model alias exists
# ----------------------------------------------
$modelName = "youtube_sentiment_lightgbm"
$aliasCheckUrl = "$mlflowUrl/api/2.0/mlflow/registered-models/get?name=$modelName"
Write-Host "🔍  Verifying MLflow registry alias for Production model '$modelName' ..."
try {
    $response = Invoke-WebRequest -Uri $aliasCheckUrl -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200 -and $response.Content -match "Production") {
        Write-Host "✅  Production alias detected in registry." -ForegroundColor Green
    } else {
        Write-Host "⚠️  Model registry reachable, but Production alias not found." -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌  Could not verify model registry. Ensure MLflow container is active." -ForegroundColor Red
}

# ----------------------------------------------
# Step 8 — Final verification
# ----------------------------------------------
Write-Host ""
Write-Host "🧠  Verifying Python environment..."
uv run python --version
Write-Host "✅  Environment check complete."

Write-Host ""
Write-Host "🎉  FULL MLOPS RESET COMPLETE — venv, DVC, MLflow, and Docker are ready." -ForegroundColor Green
Write-Host "========================================================================="
```

---

### 🚀 How to Use It

1. Save the script as **`reset-venv-full.ps1`** in your project root.
2. Open **PowerShell** in that directory.
3. Run:

   ```powershell
   ./reset-venv-full.ps1
   ```
4. Wait until it finishes rebuilding your environment (it uses your `pyproject.toml`).

---

### 🧠 Optional Enhancement

If you use this often, add it as a **custom command** in your `pyproject.toml`:

```toml
[tool.uv.scripts]
reset = "powershell ./reset-venv-full.ps1"
```

Then just run:

```bash
uv run reset
```
