#!/bin/bash
# Script to implement data refresh mechanism

echo "========================================="
echo "ECHELON: IMPLEMENTING DATA REFRESH"
echo "========================================="

# Create required directories
mkdir -p logs cron patches

# Check if patch files exist, copy them if not
if [ ! -f "patches/api_data_refresh.py" ]; then
    echo "Copying patch files..."
    cp -f patches/api_data_refresh.py patches/
fi

if [ ! -f "refresh_data.sh" ]; then
    echo "Creating data refresh script..."
    cp -f refresh_data.sh ./
    chmod +x refresh_data.sh
fi

# Apply patches to API files
echo "Applying data refresh patches to API files..."
python3 apply_data_refresh_patch.py

# Setup cron job for automated refresh
echo "Setting up scheduled data refresh..."
cron/setup_cron_job.sh

# Create trigger file for first load
touch .reload_data

echo "========================================="
echo "Data refresh implementation complete!"
echo "========================================="
echo ""
echo "The following improvements have been made:"
echo " - Added data refresh script (refresh_data.sh)"
echo " - Patched API files to support runtime data reloading"
echo " - Set up hourly cron job for automatic data refresh"
echo " - Added signal handler (SIGUSR1) for manual refresh"
echo " - Created file-based triggering mechanism"
echo ""
echo "To manually refresh data, run:"
echo "  ./refresh_data.sh"
echo ""
echo "For immediate reload in a running API server, run:"
echo "  kill -SIGUSR1 \$(pgrep -f 'python.*api')"
echo ""
echo "Check logs in the logs/ directory for refresh details."
