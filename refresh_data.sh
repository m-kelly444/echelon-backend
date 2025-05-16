#!/bin/bash

# Set environment variables and paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."
source config.sh
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="logs/refresh_$TIMESTAMP.log"

# Function to log messages
log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting data refresh process"

# Check if previous refresh is still running
if [ -f .refresh_lock ]; then
  PID=$(cat .refresh_lock)
  if ps -p $PID > /dev/null 2>&1; then
    log "Another refresh process is running (PID: $PID). Exiting."
    exit 1
  else
    log "Stale lock file found. Removing."
    rm -f .refresh_lock
  fi
fi

# Create lock file
echo $$ > .refresh_lock

# 1. Collect new data
log "Collecting new threat intelligence data..."
./scripts/collect_data.sh >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
  log "ERROR: Data collection failed."
else
  log "Data collection completed successfully."
fi

# 2. Process collected data
log "Processing collected data..."
./scripts/process_data.sh >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
  log "ERROR: Data processing failed."
else
  log "Data processing completed successfully."
fi

# 3. Signal API to reload data
# First check if the API is running
API_PID=$(pgrep -f "python.*api_server.py|python.*real_api.py" || echo "")
if [ -z "$API_PID" ]; then
  log "API server not running. No reload necessary."
else
  log "Sending reload signal to API server..."
  # Touch a reload trigger file that API can check
  touch .reload_data
  
  # For Flask-based APIs, we can use SIGUSR1 signal for reload
  if ps -p $API_PID > /dev/null 2>&1; then
    kill -SIGUSR1 $API_PID
    log "Sent reload signal to API (PID: $API_PID)"
  fi
fi

# Remove lock file
rm -f .refresh_lock

log "Data refresh process completed"

# Clean up old log files (keep last 30 days)
find logs -name "refresh_*.log" -type f -mtime +30 -delete

# Return success
exit 0
