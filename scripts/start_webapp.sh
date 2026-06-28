#!/bin/bash
# Start the Cancer Mutation Pathogenicity Predictor full stack
# (FastAPI backend + Streamlit dashboard)

set -e

API_PORT="${API_PORT:-8001}"
WEBAPP_PORT="${WEBAPP_PORT:-8501}"

echo "=================================================="
echo "  Cancer Mutation Pathogenicity Predictor"
echo "  Starting full stack..."
echo "=================================================="

echo ""
echo "Starting API server on port ${API_PORT}..."
python -m uvicorn api.main:app --host 0.0.0.0 --port "${API_PORT}" &
API_PID=$!

cleanup() {
    echo ""
    echo "Shutting down..."
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

sleep 3

echo "API server started (PID: ${API_PID})"
echo ""
echo "Starting Streamlit dashboard on port ${WEBAPP_PORT}..."
echo ""
echo "  API:       http://localhost:${API_PORT}"
echo "  Dashboard: http://localhost:${WEBAPP_PORT}"
echo "  API Docs:  http://localhost:${API_PORT}/docs"
echo ""
echo "Press Ctrl+C to stop both services."
echo "=================================================="

API_URL="http://localhost:${API_PORT}" \
    python -m streamlit run webapp/app.py \
    --server.port "${WEBAPP_PORT}" \
    --server.address 0.0.0.0
