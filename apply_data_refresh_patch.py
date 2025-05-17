import os
import re
import sys

def patch_real_api():
                                                            
    if not os.path.exists('real_api.py'):
        print("real_api.py not found, skipping.")
        return False
        
    with open('real_api.py', 'r') as f:
        content = f.read()

    if 'DataRefreshHandler' in content:
        print("real_api.py already patched for data refresh.")
        return True

    import_patch = """#!/usr/bin/env python3
import os
import json
import csv
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import glob
import signal
import threading
import time
passif __name__ == '__main__':
    # Setup data refresh handler
    refresh_handler = DataRefreshHandler(reload_data, check_interval=60)
    refresh_handler.setup_signal_handler()
    refresh_handler.start_monitoring()
    
    # Use port 5050 to avoid conflicts
    app.run(host='0.0.0.0', port=5050)
pass#!/usr/bin/env python3
import os
import json
import csv
from flask import Flask, request, jsonify
from flask_cors import CORS
import signal
import threading
import time
passif __name__ == '__main__':
    # Setup data refresh handler
    refresh_handler = DataRefreshHandler(reload_data, check_interval=60)
    refresh_handler.setup_signal_handler()
    refresh_handler.start_monitoring()
    
    # Use port 5050 to avoid conflicts
    app.run(host='0.0.0.0', port=5050)
passimport os
import json
import pickle
import re
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import signal
import threading
import time
pass
# Function to reload model data
def reload_data():
    global model, model_metadata, stats
    print("Reloading model data...")
    try:
        with open("models/threat_model.pkl", "rb") as f:
            new_model = pickle.load(f)
        
        with open("models/model_metadata.json", "r") as f:
            new_metadata = json.load(f)
        
        # Update globals
        model = new_model
        model_metadata = new_metadata
        stats["reloads"] = stats.get("reloads", 0) + 1
        stats["last_reload"] = datetime.now().isoformat()
        
        print(f"Reloaded model trained on {model_metadata.get('training_date')}")
        print(f"Model accuracy: {model_metadata.get('accuracy'):.4f}")
        return True
    except Exception as e:
        print(f"Error reloading model: {e}")
        stats["reload_errors"] = stats.get("reload_errors", 0) + 1
        return False

# Signal handler for data reload
def handle_reload_signal(signum, frame):
    if signum == signal.SIGUSR1:
        print("Received reload signal, refreshing model data...")
        reload_data()

# Register signal handler
signal.signal(signal.SIGUSR1, handle_reload_signal)

# Background thread to check for reload file
def check_reload_file():
    last_mtime = 0
    while True:
        try:
            if os.path.exists(".reload_data"):
                mtime = os.path.getmtime(".reload_data")
                if mtime > last_mtime:
                    print("Reload file modified, refreshing model data...")
                    reload_data()
                    last_mtime = mtime
        except Exception as e:
            print(f"Error checking reload file: {e}")
        
        time.sleep(60)  # Check every minute

# Start background thread for reload checks
reload_thread = threading.Thread(target=check_reload_file)
reload_thread.daemon = True
reload_thread.start()
"""

    content = re.sub(r'# Tracking stats\nstats = \{\n    "start_time": datetime\.now\(\)\.isoformat\(\),\n    "requests": 0,\n    "predictions": 0,\n    "errors": 0\n\}', '# Tracking stats\nstats = {\n    "start_time": datetime.now().isoformat(),\n    "requests": 0,\n    "predictions": 0,\n    "errors": 0,\n    "reloads": 0,\n    "reload_errors": 0\n}', content)

    pattern = 'except Exception as e:\n    print\\(f"Error loading model: {e}"\\)\n    model = None\n    model_metadata = \\{\\}'
    content = re.sub(pattern, 'except Exception as e:\n    print(f"Error loading model: {e}")\n    model = None\n    model_metadata = {}\n\n' + reload_func, content)

    with open('api_server.py', 'w') as f:
        f.write(content)
    
    print("Applied data refresh patches to api_server.py")
    return True

cat > cron/setup_cron_job.sh << 'EOF'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REFRESH_SCRIPT="$SCRIPT_DIR/../refresh_data.sh"

if [ ! -f "$REFRESH_SCRIPT" ]; then
    echo "Error: Refresh script not found at $REFRESH_SCRIPT"
    exit 1
fi

chmod +x "$REFRESH_SCRIPT"

TEMP_CRON=$(mktemp)

crontab -l > "$TEMP_CRON" 2>/dev/null || echo "" > "$TEMP_CRON"

if grep -q "$REFRESH_SCRIPT" "$TEMP_CRON"; then
    echo "Cron job for data refresh already exists."
else
                                           
    echo "# Echelon data refresh - every hour" >> "$TEMP_CRON"
    echo "0 * * * * $REFRESH_SCRIPT >> $SCRIPT_DIR/../logs/cron_refresh.log 2>&1" >> "$TEMP_CRON"

    crontab "$TEMP_CRON"
    echo "Cron job installed. Data will refresh every hour."
fi

rm "$TEMP_CRON"
