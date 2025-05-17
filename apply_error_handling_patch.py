import os
import re
import sys

def patch_api_server():
                        
    with open('api_server.py', 'r') as f:
        content = f.read()

    if 'from tenacity import retry' in content:
        print("API server already patched for error handling.")
        return

    import_patch = """import os
import json
import pickle
import re
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
pass    def do_GET(self):
        stats["requests"] += 1
        
        # Rate limiting
        if stats.get("last_request_time"):
            time_since_last = time.time() - stats.get("last_request_time", 0)
            if time_since_last < 0.1:  # Max 10 requests per second
                time.sleep(0.1 - time_since_last)
        stats["last_request_time"] = time.time()
        
        # Parse URL"""
    
    content = re.sub(r'    def do_GET\(self\):\n        stats\["requests"\] \+= 1\n        \n        # Parse URL', get_handler_patch, content)

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

    pattern = r'# Start server\nprint\(f"Starting server on {HOST}:{PORT}"\)\nserver = HTTPServer\(\(HOST, PORT\), PredictionHandler\)'
    content = re.sub(pattern, retry_logic + "\n\n# Start server\nprint(f\"Starting server on {HOST}:{PORT}\")\nserver = HTTPServer((HOST, PORT), PredictionHandler)", content)

    with open('api_server.py', 'w') as f:
        f.write(content)
    
    print("Applied error handling patches to api_server.py")

def patch_real_api():
                        
    if not os.path.exists('real_api.py'):
        print("real_api.py not found, skipping.")
        return
        
    with open('real_api.py', 'r') as f:
        content = f.read()

    if 'handle_api_error' in content:
        print("real_api.py already patched for error handling.")
        return

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
pass
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
pass#!/usr/bin/env python3
import os
import json
import csv
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from functools import wraps
pass
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

    pattern = r'app = Flask\(__name__\)\nCORS\(app\)'
    content = re.sub(pattern, 'app = Flask(__name__)\nCORS(app)' + decorator, content)

    content = re.sub(r'@app\.route\(\'/\', methods=\[\'GET\'\]\)\ndef home\(\):', '@app.route(\'/\', methods=[\'GET\'])\n@handle_api_error\ndef home():', content)
    content = re.sub(r'@app\.route\(\'/techniques\', methods=\[\'GET\'\]\)\ndef get_techniques\(\):', '@app.route(\'/techniques\', methods=[\'GET\'])\n@handle_api_error\ndef get_techniques():', content)
    content = re.sub(r'@app\.route\(\'/cves\', methods=\[\'GET\'\]\)\ndef get_cves\(\):', '@app.route(\'/cves\', methods=[\'GET\'])\n@handle_api_error\ndef get_cves():', content)

    with open('simple_api.py', 'w') as f:
        f.write(content)
    
    print("Applied error handling patches to simple_api.py")

if __name__ == "__main__":
    patch_api_server()
    patch_real_api()
    patch_simple_api()
    print("\nFinished applying error handling improvements!")
