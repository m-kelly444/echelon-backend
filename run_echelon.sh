#!/bin/bash
# Master script to run the entire Echelon system

echo "========================================="
echo "ECHELON THREAT PREDICTION SYSTEM"
echo "========================================="
echo ""

# Load configuration
source config.sh

# Record start time
start_time=$(date +%s)

# Step 1: Collect data
echo "Step 1: Collecting threat intelligence data..."
./scripts/collect_data.sh

# Step 2: Process data
echo "Step 2: Processing collected data..."
./scripts/process_data.sh

# Step 3: Train model
echo "Step 3: Training prediction model..."
./scripts/train_model.sh

# Step 4: Start API server
echo "Step 4: Starting prediction API server..."
./scripts/prediction_api.sh

# Calculate elapsed time
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total build time: $elapsed seconds"
