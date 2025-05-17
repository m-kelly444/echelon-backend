                      
import os
import sys
import re

def patch_api_server():
                                                                   
    if not os.path.exists('api_server.py'):
        print("api_server.py not found!")
        return False

    with open('api_server.py', 'r') as f:
        content = f.read()

    if 'EnhancedPredictionEngine' in content:
        print("api_server.py already enhanced!")
        return True

    imports = """import os
import json
import pickle
import re
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from models.enhanced_prediction import EnhancedPredictionEngine
pass# Initialize enhanced prediction engine
try:
    prediction_engine = EnhancedPredictionEngine()
    print("Initialized enhanced prediction engine")
except Exception as e:
    print(f"Error initializing enhanced prediction engine: {e}")
    prediction_engine = None
pass                # Get CVE ID if provided
                cve_id = data.get("cve_id", "")
                
                # Use enhanced prediction engine if available
                if prediction_engine:
                    try:
                        # Make enhanced prediction
                        enhanced_result = prediction_engine.predict([cve_year, base_score, days_since_published], cve_id=cve_id)
                        
                        # The enhanced result already includes all the basic prediction data
                        response = enhanced_result
                    except Exception as e:
                        print(f"Error using enhanced prediction: {e}")
                        # Fall back to basic prediction below
                        response = {
                            "error": f"Enhanced prediction failed: {str(e)}",
                            "fallback": "Using basic prediction"
                        }
                else:
                    # Make basic prediction with the original model
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
                            "base_score": base_score,
                            "cve_id": cve_id
                        }
                    }"""

    pattern = '                # Make prediction\n                prediction_proba = model.predict_proba\\(\\[\\[cve_year, base_score\\]\\]\\)\\[0\\]\n                prediction = int\\(model.predict\\(\\[\\[cve_year, base_score\\]\\]\\)\\[0\\]\\)\n                threat_score = float\\(prediction_proba\\[1\\]\\)\n                \n                # Determine threat level\n                threat_level = "LOW"\n                if threat_score > 0\\.7:\n                    threat_level = "HIGH"\n                elif threat_score > 0\\.4:\n                    threat_level = "MEDIUM"\n                \n                # Prepare response\n                response = \\{\n                    "prediction": prediction,\n                    "threat_score": round\\(threat_score, 4\\),\n                    "threat_level": threat_level,\n                    "confidence": round\\(abs\\(prediction_proba\\[1\\] - prediction_proba\\[0\\]\\), 4\\),\n                    "timestamp": datetime\\.now\\(\\)\\.isoformat\\(\\),\n                    "input_features": \\{\n                        "cve_year": cve_year,\n                        "base_score": base_score\n                    \\}\n                \\}'
    
    content = re.sub(pattern, prediction_code, content)

    features_code = """                # Feature 1: CVE year
                cve_year = 2020  # Default
                if "cve_id" in data and re.match(r"CVE-\\d+-\\d+", data["cve_id"]):
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
                
                # Feature 3: Days since published (default to 0 for new vulnerabilities)
                days_since_published = 0
                if "published_date" in data:
                    try:
                        published = datetime.fromisoformat(data["published_date"].replace("Z", "+00:00"))
                        days_since_published = (datetime.now() - published).days
                    except:
                        pass"""
    
    pattern = '                # Feature 1: CVE year\n                cve_year = 2020  # Default\n                if "cve_id" in data and re.match\\(r"CVE-\\\\d\\+-\\\\d\\+", data\\["cve_id"\\]\\):\n                    try:\n                        cve_year = int\\(data\\["cve_id"\\]\\.split\\("-"\\)\\[1\\]\\)\n                    except:\n                        pass\n                elif "cve_year" in data:\n                    cve_year = int\\(data\\["cve_year"\\]\\)\n                \n                # Feature 2: CVSS Base Score\n                base_score = 5\\.0  # Default\n                if "base_score" in data:\n                    base_score = float\\(data\\["base_score"\\]\\)\n                elif "severity" in data:\n                    base_score = float\\(data\\["severity"\\]\\)'
    
    content = re.sub(pattern, features_code, content)

    geo_endpoint = """        # Geographic data endpoint
        elif path == "/geo":
            self._set_headers()
            
            try:
                # Check if we have processed geographic data
                if os.path.exists("data/processed/geo/threat_locations.json"):
                    with open("data/processed/geo/threat_locations.json", "r") as f:
                        geo_data = json.load(f)
                elif os.path.exists("data/processed/geo/abuse_locations.json"):
                    with open("data/processed/geo/abuse_locations.json", "r") as f:
                        geo_data = json.load(f)
                else:
                    geo_data = []
                
                # Return the data
                response = {
                    "count": len(geo_data),
                    "locations": geo_data
                }
            except Exception as e:
                response = {
                    "error": f"Error loading geographic data: {str(e)}",
                    "locations": []
                }
            
            self.wfile.write(json.dumps(response).encode())
            return"""

    pattern = '        # Not found\n        else:'
    content = re.sub(pattern, geo_endpoint + '\n\n        # Not found\n        else:', content)

    home_endpoints = '                    {"path": "/", "method": "GET", "description": "API information"},\n                    {"path": "/predict", "method": "POST", "description": "Make a prediction"},\n                    {"path": "/cves", "method": "GET", "description": "List processed CVEs"},\n                    {"path": "/geo", "method": "GET", "description": "Get geographic threat data"},\n                    {"path": "/status", "method": "GET", "description": "Check API status"}'
    
    pattern = '                    \\{"path": "/", "method": "GET", "description": "API information"\\},\n                    \\{"path": "/predict", "method": "POST", "description": "Make a prediction"\\},\n                    \\{"path": "/cves", "method": "GET", "description": "List processed CVEs"\\},\n                    \\{"path": "/status", "method": "GET", "description": "Check API status"\\}'
    
    content = re.sub(pattern, home_endpoints, content)

    with open('api_server.py', 'w') as f:
        f.write(content)
    
    print("Successfully patched api_server.py with enhanced prediction capabilities!")
    return True

if __name__ == "__main__":
    patch_api_server()
