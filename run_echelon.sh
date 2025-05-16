#!/bin/bash
# run_echelon.sh - Run Echelon with real data only

echo "========================================="
echo "ECHELON THREAT PREDICTION SYSTEM"
echo "========================================="
echo "REAL DATA ONLY MODE"
echo "========================================="

# Check for real data requirements
echo "Checking real data requirements..."
python3 check_real_data.py
if [ $? -ne 0 ]; then
    echo "ERROR: Required real data is missing."
    echo "Run enhance_backend.sh to collect real threat intelligence data."
    exit 1
fi

# Start API server
echo "Starting API server with real data..."
python3 api_server.py
