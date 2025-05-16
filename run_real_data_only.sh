#!/bin/bash
# run_real_data_only.sh - Run Echelon with only real data sources, no fallbacks

echo "========================================="
echo "ECHELON: REAL DATA ONLY MODE"
echo "========================================="

# Validate real data is available
echo "Validating real data sources..."
python3 validate_real_data.py
if [ $? -ne 0 ]; then
  echo "ERROR: Real data validation failed."
  echo "Please run data collection scripts with valid API keys first."
  echo "You can use: ./enhance_backend.sh"
  exit 1
fi

# Start the API server with real data only
echo "Starting API server with real data only..."
python3 api_server.py
