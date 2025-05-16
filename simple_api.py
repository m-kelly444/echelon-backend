#!/usr/bin/env python3
import os
import json
import csv
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load real data only
def load_real_data():
    data = {"techniques": [], "cves": []}
    
    # Load MITRE ATT&CK data
    if os.path.exists('data/mitre/enterprise_attack.json'):
        try:
            with open('data/mitre/enterprise_attack.json', 'r') as f:
                attack_data = json.load(f)
                
            for obj in attack_data.get('objects', []):
                if obj.get('type') == 'attack-pattern':
                    # Get technique ID
                    tech_id = ""
                    for ref in obj.get('external_references', []):
                        if ref.get('source_name') == 'mitre-attack':
                            tech_id = ref.get('external_id', '')
                            break
                    
                    if tech_id:
                        data["techniques"].append({
                            "id": tech_id,
                            "name": obj.get('name', ''),
                            "description": obj.get('description', '')
                        })
            print(f"Loaded {len(data['techniques'])} ATT&CK techniques")
        except Exception as e:
            print(f"Error loading ATT&CK data: {e}")
    
    # Load CISA KEV data
    if os.path.exists('data/api/cisa_kev.csv'):
        try:
            with open('data/api/cisa_kev.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Get CVE ID column (might have different names)
                    cve_id = next((row[key] for key in row.keys() 
                                  if 'cve' in key.lower() and 'id' in key.lower()), None)
                    
                    if cve_id:
                        data["cves"].append(row)
            print(f"Loaded {len(data['cves'])} CVEs from CISA KEV")
        except Exception as e:
            print(f"Error loading CISA KEV data: {e}")
    
    return data

# Load data at startup
real_data = load_real_data()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "name": "Echelon Real Data API",
        "data_available": {
            "techniques": len(real_data["techniques"]),
            "cves": len(real_data["cves"])
        }
    })

@app.route('/techniques', methods=['GET'])
def get_techniques():
    # Get pagination parameters
    limit = min(int(request.args.get('limit', 50)), 500)
    offset = int(request.args.get('offset', 0))
    
    # Return real techniques with pagination
    return jsonify({
        "total": len(real_data["techniques"]),
        "techniques": real_data["techniques"][offset:offset+limit]
    })

@app.route('/cves', methods=['GET'])
def get_cves():
    # Get pagination parameters
    limit = min(int(request.args.get('limit', 50)), 500)
    offset = int(request.args.get('offset', 0))
    
    # Return real CVEs with pagination
    return jsonify({
        "total": len(real_data["cves"]),
        "cves": real_data["cves"][offset:offset+limit]
    })

if __name__ == '__main__':
    # Use port 8080 to avoid conflicts with macOS AirPlay (port 5000)
    app.run(host='0.0.0.0', port=8080)
