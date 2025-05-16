#!/bin/bash
# Script to start the prediction API server

source config.sh

echo "Starting Echelon Prediction API at $(date)"

# Use Python for the API server
python3 -c '
import os
import json
import pickle
import time
import re
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Load configuration
API_PORT = 8080
API_HOST = "0.0.0.0"

# Load model
try:
    with open("models/threat_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("models/model_metadata.json", "r") as f:
        model_metadata = json.load(f)
    
    print(f"Loaded model trained on {model_metadata.get(\"training_date\")}")
    print(f"Model accuracy: {model_metadata.get(\"accuracy\"):.4f}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("API will run without prediction capabilities")
    model = None
    model_metadata = {}

# Statistics
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
        
        # Parse URL and query parameters
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
                    {"path": "/cves", "method": "GET", "description": "List processed CVEs"},
                    {"path": "/status", "method": "GET", "description": "Check API status"}
                ]
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
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
            return
        
        # CVEs endpoint
        elif path == "/cves":
            self._set_headers()
            
            # Get pagination parameters
            query = urllib.parse.parse_qs(parsed_path.query)
            limit = int(query.get("limit", ["50"])[0])
            offset = int(query.get("offset", ["0"])[0])
            
            # Load CVEs
            cves = []
            cve_dir = "data/processed/cves"
            if os.path.exists(cve_dir):
                files = os.listdir(cve_dir)
                for i, cve_file in enumerate(files[offset:offset+limit]):
                    if cve_file.endswith(".json"):
                        try:
                            with open(os.path.join(cve_dir, cve_file), "r") as f:
                                cve = json.load(f)
                                cves.append(cve)
                        except Exception as e:
                            print(f"Error loading {cve_file}: {e}")
            
            response = {
                "total": len(os.listdir(cve_dir)) if os.path.exists(cve_dir) else 0,
                "limit": limit,
                "offset": offset,
                "cves": cves
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
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
                    self.wfile.write(json.dumps({"error": "Model not loaded"}).encode())
                    stats["errors"] += 1
                    return
                
                # Extract features
                features = []
                
                # Feature 1: CVE year
                cve_year = 2020  # Default
                if "cve_id" in data and re.match(r"CVE-\d+-\d+", data["cve_id"]):
                    try:
                        cve_year = int(data["cve_id"].split("-")[1])
                    except:
                        pass
                elif "cve_year" in data:
                    cve_year = int(data["cve_year"])
                features.append(cve_year)
                
                # Feature 2: CVSS Base Score
                base_score = data.get("base_score", data.get("severity", 5.0))
                features.append(float(base_score))
                
                # Feature 3: Days since published (default to 0 for new vulnerabilities)
                features.append(0)
                
                # Make prediction
                prediction_proba = model.predict_proba([features])[0]
                prediction = int(model.predict([features])[0])
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

# Start server
print(f"Starting server on {API_HOST}:{API_PORT}")
server = HTTPServer((API_HOST, API_PORT), PredictionHandler)

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("Server stopped by user")
    server.server_close()
'

echo "API server running at http://$API_HOST:$API_PORT"
