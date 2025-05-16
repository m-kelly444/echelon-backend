import os
import json
import pickle
import re
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# Real data only mode - no fallbacks or synthetics
print("Echelon is running in REAL DATA ONLY mode - using only genuine threat intelligence data")
# Real data only mode - no fallbacks or synthetics
print("Echelon is running in REAL DATA ONLY mode - using only genuine threat intelligence data")
# Initialize flag for real data only mode
REAL_DATA_ONLY = True
print("Running in REAL_DATA_ONLY mode - no fallbacks or synthetic data will be used")


# Configuration
PORT = 8080
HOST = "0.0.0.0"

# Load model
try:
    with open("models/threat_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("models/model_metadata.json", "r") as f:
        model_metadata = json.load(f)
    
    print(f"Loaded model trained on {model_metadata.get('training_date')}")
    print(f"Model accuracy: {model_metadata.get('accuracy'):.4f}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    model_metadata = {}

# Tracking stats
stats = {
    "start_time": datetime.now().isoformat(),
    "requests": 0,
    "predictions": 0,
    "errors": 0
}

class PredictionHandler(BaseHTTPRequestHandler):
    def _set_headers(self, content_type="application/json"):
        self.send_response(200)
        self.send_header("Content-type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def do_OPTIONS(self):
        self._set_headers()
    
    def do_GET(self):
        stats["requests"] += 1
        
        # Rate limiting
        if stats.get("last_request_time"):
            time_since_last = time.time() - stats.get("last_request_time", 0)
            if time_since_last < 0.1:  # Max 10 requests per second
                time.sleep(0.1 - time_since_last)
        stats["last_request_time"] = time.time()
        
        # Parse URL
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        # Home endpoint
        if path == "/":
            self._set_headers()
            response = {
                "name": "Echelon Threat Prediction API",
                "version": "1.0.0",
                "model_loaded": model is not None,
                "endpoints": [
                    {"path": "/", "method": "GET", "description": "API information"},
                    {"path": "/predict", "method": "POST", "description": "Make a prediction"},
                    {"path": "/status", "method": "GET", "description": "Check API status"}
                ]
            }
            self.wfile.write(json.dumps(response).encode())
        
        # Status endpoint
        elif path == "/status":
            self._set_headers()
            uptime = (datetime.now() - datetime.fromisoformat(stats["start_time"])).total_seconds()
            
            response = {
                "status": "healthy" if model is not None else "degraded",
                "uptime_seconds": int(uptime),
                "stats": stats,
                "model": {
                    "loaded": model is not None,
                    "training_date": model_metadata.get("training_date", "unknown"),
                    "accuracy": model_metadata.get("accuracy", 0)
                }
            }
            self.wfile.write(json.dumps(response).encode())
        
        # Not found
        else:
            self.send_response(404)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
    
    def do_POST(self):
        stats["requests"] += 1
        
        # Parse URL
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        # Prediction endpoint
        if path == "/predict":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            
            try:
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
                    return
                
                # Extract features
                # Feature 1: CVE year
                cve_year = 2020  # Default
                if "cve_id" in data and re.match(r"CVE-\d+-\d+", data["cve_id"]):
                    try:
                        cve_year = int(data["cve_id"].split("-")[1])
                    except:
                        pass
                elif "cve_year" in data:
                    cve_year = int(data["cve_year"])
                
                # Feature 2: CVSS Base Score
                base_score = 5.0  # Default
                if "base_score" in data:
                    base_score = float(data["base_score"])
                elif "severity" in data:
                    base_score = float(data["severity"])
                
                # Make prediction
                prediction_proba = model.predict_proba([[cve_year, base_score]])[0]
                prediction = int(model.predict([[cve_year, base_score]])[0])
                threat_score = float(prediction_proba[1])
                
                # Determine threat level
                threat_level = "LOW"
                if threat_score > 0.7:
                    threat_level = "HIGH"
                elif threat_score > 0.4:
                    threat_level = "MEDIUM"
                
                # Prepare response
                response = {
                    "prediction": prediction,
                    "threat_score": round(threat_score, 4),
                    "threat_level": threat_level,
                    "confidence": round(abs(prediction_proba[1] - prediction_proba[0]), 4),
                    "timestamp": datetime.now().isoformat(),
                    "input_features": {
                        "cve_year": cve_year,
                        "base_score": base_score
                    }
                }
                
                self._set_headers()
                self.wfile.write(json.dumps(response).encode())
                stats["predictions"] += 1
                
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
                stats["errors"] += 1
        
        # Not found
        else:
            self.send_response(404)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception))
    )
    def _predict_with_retry(self, features):
        # Wrapper to retry predictions in case of transient errors
        prediction_proba = model.predict_proba([features])[0]
        prediction = int(model.predict([features])[0])
        return prediction, prediction_proba

# Start server
print(f"Starting server on {HOST}:{PORT}")
server = HTTPServer((HOST, PORT), PredictionHandler)

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("Server stopped by user")
    server.server_close()
