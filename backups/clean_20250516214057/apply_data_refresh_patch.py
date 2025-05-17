import os
import re
import sys

def patch_real_api():
    """Patch real_api.py to add data refresh capabilities"""
    if not os.path.exists('real_api.py'):
        print("real_api.py not found, skipping.")
        return False
        
    with open('real_api.py', 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'DataRefreshHandler' in content:
        print("real_api.py already patched for data refresh.")
        return True
    
    # Add imports
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
"""
    
    content = re.sub(r'#!/usr/bin/env python3\nimport os\nimport json\nimport csv\nimport xml\.etree\.ElementTree as ET\nfrom flask import Flask, request, jsonify\nfrom flask_cors import CORS\nfrom datetime import datetime\nimport glob', import_patch, content)
    
    # Add the DataRefreshHandler class
    with open('patches/api_data_refresh.py', 'r') as f:
        refresh_handler_code = f.read()
    
    # Insert the class after imports
    pattern = 'app = Flask\\(__name__\\)'
    content = re.sub(pattern, refresh_handler_code + '\n\napp = Flask(__name__)', content)
    
    # Modify the load_all_data function to be reusable for refresh
    content = re.sub(r'# Load all data at startup\nall_data = load_all_data\\(\\)', '# Create a function to reload data\ndef reload_data():\n    global all_data\n    print("Reloading data from sources...")\n    all_data = load_all_data()\n    print(f"Data refresh complete. Loaded {len(all_data.get(\'cves\', []))} CVEs, " \\\n          f"{len(all_data.get(\'techniques\', []))} techniques, and " \\\n          f"{len(all_data.get(\'alerts\', []))} alerts.")\n\n# Load all data at startup\nall_data = load_all_data()', content)
    
    # Add the signal handler and monitor setup at the end
    main_block = """if __name__ == '__main__':
    # Setup data refresh handler
    refresh_handler = DataRefreshHandler(reload_data, check_interval=60)
    refresh_handler.setup_signal_handler()
    refresh_handler.start_monitoring()
    
    # Use port 5050 to avoid conflicts
    app.run(host='0.0.0.0', port=5050)
"""
    
    content = re.sub(r'if __name__ == \'__main__\':\n    # Use port [0-9]+ to avoid conflicts.*\n    app\.run\(host=\'0\.0\.0\.0\', port=[0-9]+\)', main_block, content)
    
    # Write updated content
    with open('real_api.py', 'w') as f:
        f.write(content)
    
    print("Applied data refresh patches to real_api.py")
    return True

def patch_simple_api():
    """Patch simple_api.py to add data refresh capabilities"""
    if not os.path.exists('simple_api.py'):
        print("simple_api.py not found, skipping.")
        return False
        
    with open('simple_api.py', 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'DataRefreshHandler' in content:
        print("simple_api.py already patched for data refresh.")
        return True
    
    # Add imports
    import_patch = """#!/usr/bin/env python3
import os
import json
import csv
from flask import Flask, request, jsonify
from flask_cors import CORS
import signal
import threading
import time
"""
    
    content = re.sub(r'#!/usr/bin/env python3\nimport os\nimport json\nimport csv\nfrom flask import Flask, request, jsonify\nfrom flask_cors import CORS', import_patch, content)
    
    # Add the DataRefreshHandler class
    with open('patches/api_data_refresh.py', 'r') as f:
        refresh_handler_code = f.read()
    
    # Insert the class after imports
    pattern = 'app = Flask\\(__name__\\)'
    content = re.sub(pattern, refresh_handler_code + '\n\napp = Flask(__name__)', content)
    
    # Modify the load_real_data function to be reusable for refresh
    content = re.sub(r'# Load data at startup\nreal_data = load_real_data\\(\\)', '# Create a function to reload data\ndef reload_data():\n    global real_data\n    print("Reloading data from sources...")\n    real_data = load_real_data()\n    print(f"Data refresh complete. Loaded {len(real_data.get(\'cves\', []))} CVEs and " \\\n          f"{len(real_data.get(\'techniques\', []))} techniques.")\n\n# Load data at startup\nreal_data = load_real_data()', content)
    
    # Add the signal handler and monitor setup at the end
    main_block = """if __name__ == '__main__':
    # Setup data refresh handler
    refresh_handler = DataRefreshHandler(reload_data, check_interval=60)
    refresh_handler.setup_signal_handler()
    refresh_handler.start_monitoring()
    
    # Use port 5050 to avoid conflicts
    app.run(host='0.0.0.0', port=5050)
"""
    
    content = re.sub(r'if __name__ == \'__main__\':\n    # Use port [0-9]+ to avoid conflicts.*\n    app\.run\(host=\'0\.0\.0\.0\', port=[0-9]+\)', main_block, content)
    
    # Write updated content
    with open('simple_api.py', 'w') as f:
        f.write(content)
    
    print("Applied data refresh patches to simple_api.py")
    return True

def patch_api_server():
    """Patch api_server.py to add data refresh capabilities"""
    if not os.path.exists('api_server.py'):
        print("api_server.py not found, skipping.")
        return False
        
    with open('api_server.py', 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'reload_data' in content:
        print("api_server.py already patched for data refresh.")
        return True
    
    # Add imports
    import_patch = """import os
import json
import pickle
import re
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import signal
import threading
import time
"""
    
    content = re.sub(r'import os\nimport json\nimport pickle\nimport re\nfrom datetime import datetime\nfrom http\.server import HTTPServer, BaseHTTPRequestHandler\nimport urllib\.parse', import_patch, content)
    
    # Add function to reload the model
    reload_func = """
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
    
    # Add reload stats to the stats dictionary
    content = re.sub(r'# Tracking stats\nstats = \{\n    "start_time": datetime\.now\(\)\.isoformat\(\),\n    "requests": 0,\n    "predictions": 0,\n    "errors": 0\n\}', '# Tracking stats\nstats = {\n    "start_time": datetime.now().isoformat(),\n    "requests": 0,\n    "predictions": 0,\n    "errors": 0,\n    "reloads": 0,\n    "reload_errors": 0\n}', content)
    
    # Insert reload function after model loading
    pattern = 'except Exception as e:\n    print\\(f"Error loading model: {e}"\\)\n    model = None\n    model_metadata = \\{\\}'
    content = re.sub(pattern, 'except Exception as e:\n    print(f"Error loading model: {e}")\n    model = None\n    model_metadata = {}\n\n' + reload_func, content)
    
    # Write updated content
    with open('api_server.py', 'w') as f:
        f.write(content)
    
    print("Applied data refresh patches to api_server.py")
    return True

# Create cron job to run the refresh script
cat > cron/setup_cron_job.sh << 'EOF'
#!/bin/bash

# Get absolute path to the refresh script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REFRESH_SCRIPT="$SCRIPT_DIR/../refresh_data.sh"

# Check if script exists
if [ ! -f "$REFRESH_SCRIPT" ]; then
    echo "Error: Refresh script not found at $REFRESH_SCRIPT"
    exit 1
fi

# Make sure script is executable
chmod +x "$REFRESH_SCRIPT"

# Create a temporary file for the crontab
TEMP_CRON=$(mktemp)

# Export current crontab
crontab -l > "$TEMP_CRON" 2>/dev/null || echo "" > "$TEMP_CRON"

# Check if entry already exists
if grep -q "$REFRESH_SCRIPT" "$TEMP_CRON"; then
    echo "Cron job for data refresh already exists."
else
    # Add our cron job - refresh every hour
    echo "# Echelon data refresh - every hour" >> "$TEMP_CRON"
    echo "0 * * * * $REFRESH_SCRIPT >> $SCRIPT_DIR/../logs/cron_refresh.log 2>&1" >> "$TEMP_CRON"
    
    # Install the new crontab
    crontab "$TEMP_CRON"
    echo "Cron job installed. Data will refresh every hour."
fi

# Clean up
rm "$TEMP_CRON"
