@echo off
REM Start the Cancer Mutation Pathogenicity Predictor full stack
REM (FastAPI backend + Streamlit dashboard)

echo ==================================================
echo   Cancer Mutation Pathogenicity Predictor
echo   Starting full stack...
echo ==================================================

set API_PORT=8001
set WEBAPP_PORT=8501

echo.
echo Starting API server on port %API_PORT%...
start /B python -m uvicorn api.main:app --host 0.0.0.0 --port %API_PORT%

timeout /t 3 /nobreak >nul

echo API server started.
echo.
echo Starting Streamlit dashboard on port %WEBAPP_PORT%...
echo.
echo   API:       http://localhost:%API_PORT%
echo   Dashboard: http://localhost:%WEBAPP_PORT%
echo   API Docs:  http://localhost:%API_PORT%/docs
echo.
echo Press Ctrl+C to stop both services.
echo ==================================================

set API_URL=http://localhost:%API_PORT%
python -m streamlit run webapp/app.py --server.port %WEBAPP_PORT%
