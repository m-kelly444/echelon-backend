import os
import re
import sys

def patch_api_server():
    # Read original file
    with open('api_server.py', 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'from tenacity import retry' in content:
        print("API server already patched for error handling.")
        return
    
    # Add imports
    import_patch = """import os
import json
import pickle
import re
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
"""
    
    # Replace imports
    content = re.sub(r'import os\nimport json\nimport pickle\nimport re\nfrom datetime import datetime\nfrom http\.server import HTTPServer, BaseHTTPRequestHandler\nimport urllib\.parse', import_patch, content)
    
    # Add rate limiting to GET handler
    get_handler_patch = """    def do_GET(self):
        stats["requests"] += 1
        
        # Rate limiting
        if stats.get("last_request_time"):
            time_since_last = time.time() - stats.get("last_request_time", 0)
            if time_since_last < 0.1:  # Max 10 requests per second
                time.sleep(0.1 - time_since_last)
        stats["last_request_time"] = time.time()
        
        # Parse URL"""
    
    content = re.sub(r'    def do_GET\(self\):\n        stats\["requests"\] \+= 1\n        \n        # Parse URL', get_handler_patch, content)
    
    # Add better error handling to POST handler
    post_handler_patch = """            try:
                data = json.loads(post_data.decode())
                
                if model is None:
                    self.send_response(503)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "error": "Model not loaded",
                        "retry_after": 30,
                        "timestamp": datetime.now().isoformat()
                    }).encode())
                    stats["errors"] += 1
                    return"""
    
    content = re.sub(r'            try:\n                data = json.loads\(post_data.decode\(\)\)\n                \n                if model is None:\n                    self.send_response\(503\)\n                    self.send_header\("Content-type", "application/json"\)\n                    self.end_headers\(\)\n                    self.wfile.write\(json.dumps\({"error": "Model not loaded"}\).encode\(\)\)\n                    stats\["errors"\] \+= 1\n                    return', post_handler_patch, content)
    
    # Add retry logic
    retry_logic = """
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception))
    )
    def _predict_with_retry(self, features):
        # Wrapper to retry predictions in case of transient errors
        prediction_proba = model.predict_proba([features])[0]
        prediction = int(model.predict([features])[0])
        return prediction, prediction_proba"""
    
    # Insert the retry logic before the server = HTTPServer line
    pattern = r'# Start server\nprint\(f"Starting server on {HOST}:{PORT}"\)\nserver = HTTPServer\(\(HOST, PORT\), PredictionHandler\)'
    content = re.sub(pattern, retry_logic + "\n\n# Start server\nprint(f\"Starting server on {HOST}:{PORT}\")\nserver = HTTPServer((HOST, PORT), PredictionHandler)", content)
    
    # Write updated content
    with open('api_server.py', 'w') as f:
        f.write(content)
    
    print("Applied error handling patches to api_server.py")

def patch_real_api():
    # Read original file
    if not os.path.exists('real_api.py'):
        print("real_api.py not found, skipping.")
        return
        
    with open('real_api.py', 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'handle_api_error' in content:
        print("real_api.py already patched for error handling.")
        return
    
    # Add import
    import_patch = """#!/usr/bin/env python3
import os
import json
import csv
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import glob
import time
from functools import wraps
"""
    
    content = re.sub(r'#!/usr/bin/env python3\nimport os\nimport json\nimport csv\nimport xml\.etree\.ElementTree as ET\nfrom flask import Flask, request, jsonify\nfrom flask_cors import CORS\nfrom datetime import datetime\nimport glob', import_patch, content)
    
    # Add decorator
    decorator = """
# Error handling decorator
def handle_api_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the error
            print(f"API Error in {func.__name__}: {str(e)}")
            
            error_response = {
                "error": True,
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
                "endpoint": func.__name__
            }
            
            # Return appropriate status
            status_code = 500
            if isinstance(e, ValueError) or isinstance(e, KeyError):
                status_code = 400
            elif isinstance(e, FileNotFoundError):
                status_code = 404
                
            return jsonify(error_response), status_code
    return wrapper
"""
    
    # Insert after CORS setup
    pattern = r'app = Flask\(__name__\)\nCORS\(app\)'
    content = re.sub(pattern, 'app = Flask(__name__)\nCORS(app)' + decorator, content)
    
    # Apply decorator to routes
    content = re.sub(r'@app\.route\(\'/\', methods=\[\'GET\'\]\)\ndef home\(\):', '@app.route(\'/\', methods=[\'GET\'])\n@handle_api_error\ndef home():', content)
    content = re.sub(r'@app\.route\(\'/cves\', methods=\[\'GET\'\]\)\ndef get_cves\(\):', '@app.route(\'/cves\', methods=[\'GET\'])\n@handle_api_error\ndef get_cves():', content)
    content = re.sub(r'@app\.route\(\'/techniques\', methods=\[\'GET\'\]\)\ndef get_techniques\(\):', '@app.route(\'/techniques\', methods=[\'GET\'])\n@handle_api_error\ndef get_techniques():', content)
    content = re.sub(r'@app\.route\(\'/alerts\', methods=\[\'GET\'\]\)\ndef get_alerts\(\):', '@app.route(\'/alerts\', methods=[\'GET\'])\n@handle_api_error\ndef get_alerts():', content)
    
    # Write updated content
    with open('real_api.py', 'w') as f:
        f.write(content)
    
    print("Applied error handling patches to real_api.py")

def patch_simple_api():
    # Similar treatment for simple_api.py
    if not os.path.exists('simple_api.py'):
        print("simple_api.py not found, skipping.")
        return
        
    with open('simple_api.py', 'r') as f:
        content = f.read()
    
    # Check if already patched
    if 'handle_api_error' in content:
        print("simple_api.py already patched for error handling.")
        return
    
    # Add imports for error handling
    import_patch = """#!/usr/bin/env python3
import os
import json
import csv
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from functools import wraps
"""
    
    content = re.sub(r'#!/usr/bin/env python3\nimport os\nimport json\nimport csv\nfrom flask import Flask, request, jsonify\nfrom flask_cors import CORS', import_patch, content)
    
    # Add decorator
    decorator = """
# Error handling decorator
def handle_api_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the error
            print(f"API Error in {func.__name__}: {str(e)}")
            
            error_response = {
                "error": True,
                "message": str(e),
                "timestamp": time.time(),
                "endpoint": func.__name__
            }
            
            # Return appropriate status
            status_code = 500
            if isinstance(e, ValueError) or isinstance(e, KeyError):
                status_code = 400
            elif isinstance(e, FileNotFoundError):
                status_code = 404
                
            return jsonify(error_response), status_code
    return wrapper
"""
    
    # Insert after CORS setup
    pattern = r'app = Flask\(__name__\)\nCORS\(app\)'
    content = re.sub(pattern, 'app = Flask(__name__)\nCORS(app)' + decorator, content)
    
    # Apply decorator to routes
    content = re.sub(r'@app\.route\(\'/\', methods=\[\'GET\'\]\)\ndef home\(\):', '@app.route(\'/\', methods=[\'GET\'])\n@handle_api_error\ndef home():', content)
    content = re.sub(r'@app\.route\(\'/techniques\', methods=\[\'GET\'\]\)\ndef get_techniques\(\):', '@app.route(\'/techniques\', methods=[\'GET\'])\n@handle_api_error\ndef get_techniques():', content)
    content = re.sub(r'@app\.route\(\'/cves\', methods=\[\'GET\'\]\)\ndef get_cves\(\):', '@app.route(\'/cves\', methods=[\'GET\'])\n@handle_api_error\ndef get_cves():', content)
    
    # Write updated content
    with open('simple_api.py', 'w') as f:
        f.write(content)
    
    print("Applied error handling patches to simple_api.py")

# Main execution
if __name__ == "__main__":
    patch_api_server()
    patch_real_api()
    patch_simple_api()
    print("\nFinished applying error handling improvements!")
