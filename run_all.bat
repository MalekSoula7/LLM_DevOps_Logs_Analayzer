@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  DevOps Log Analyzer - Starting All Services
echo ============================================================

:: Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [OK] Virtual environment activated.
) else (
    echo [WARN] No .venv found. Attempting with system Python.
)

:: Start MLflow tracking server (port 5000) in a new window
echo [1/5] Starting MLflow server on http://127.0.0.1:5000 ...
start "MLflow Server" cmd /k "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns"

:: Wait for MLflow to become ready (polls /health, up to 30s)
echo     Waiting for MLflow to become ready...
set MLFLOW_READY=0
for /l %%i in (1,1,30) do (
    curl -s -o nul -w "%%{http_code}" http://127.0.0.1:5000/health > "%TEMP%\mlflow_health.txt" 2>nul
    set /p MLFLOW_CODE=<"%TEMP%\mlflow_health.txt"
    if "!MLFLOW_CODE!"=="200" (
        set MLFLOW_READY=1
        goto :mlflow_ready
    )
    timeout /t 1 /nobreak >nul
)
:mlflow_ready
if "!MLFLOW_READY!"=="1" (
    echo [OK] MLflow is ready.
) else (
    echo [WARN] MLflow did not respond within 30s. Check the "MLflow Server" window for errors.
)

:: Ensure the Prefect client (this script + the training pipeline) targets our local server
set PREFECT_API_URL=http://127.0.0.1:4200/api

:: Start Prefect server (port 4200) in a new window
echo [2/5] Starting Prefect server on http://127.0.0.1:4200 ...
start "Prefect Server" cmd /k "prefect server start --host 0.0.0.0 --port 4200"

:: Wait for Prefect to become ready (polls /api/health, up to 45s — first run includes DB migrations)
echo     Waiting for Prefect server to become ready...
set PREFECT_READY=0
for /l %%i in (1,1,45) do (
    curl -s -o nul -w "%%{http_code}" http://127.0.0.1:4200/api/health > "%TEMP%\prefect_health.txt" 2>nul
    set /p PREFECT_CODE=<"%TEMP%\prefect_health.txt"
    if "!PREFECT_CODE!"=="200" (
        set PREFECT_READY=1
        goto :prefect_ready
    )
    timeout /t 1 /nobreak >nul
)
:prefect_ready
if "!PREFECT_READY!"=="1" (
    echo [OK] Prefect server is ready.
) else (
    echo [WARN] Prefect server did not respond within 45s. Check the "Prefect Server" window for errors.
    echo         The training pipeline below may fail to connect.
)

:: Run the training pipeline (blocking — must complete before API starts)
echo [3/5] Running training pipeline...
python flows\training_flow.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Training pipeline failed. Check the output above.
    echo         You can still start the API/dashboard with a pre-trained model.pkl
    echo         Press any key to continue starting remaining services...
    pause >nul
)

:: Start FastAPI backend (port 8000) in a new window
echo [4/5] Starting FastAPI backend on http://localhost:8000 ...
start "FastAPI Backend" cmd /k "uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"

:: Wait for API to boot
echo     Waiting 3 seconds for FastAPI to start...
timeout /t 3 /nobreak >nul

:: Start Streamlit dashboard (port 8501) in a new window
echo [5/5] Starting Streamlit dashboard on http://localhost:8501 ...
start "Streamlit Dashboard" cmd /k "streamlit run dashboard\app.py --server.port 8501 --server.address 0.0.0.0"

echo.
echo ============================================================
echo  All services started successfully!
echo ============================================================
echo.
echo  Dashboard    : http://localhost:8501
echo  API docs     : http://localhost:8000/docs
echo  MLflow UI    : http://localhost:5000
echo  Prefect UI   : http://localhost:4200
echo.
echo  To stop all services, close the individual terminal windows
echo  or press Ctrl+C in each one.
echo ============================================================
echo.

endlocal
