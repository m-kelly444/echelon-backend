                      
import os
import json
import csv
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from functools import wraps

app = Flask(__name__)
CORS(app)
                          
def handle_api_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
                           
            print(f"API Error in {func.__name__}: {str(e)}")
            
            error_response = {
                "error": True,
                "message": str(e),
                "timestamp": time.time(),
                "endpoint": func.__name__
            }

            status_code = 500
            if isinstance(e, ValueError) or isinstance(e, KeyError):
                status_code = 400
            elif isinstance(e, FileNotFoundError):
                status_code = 404
                
            return jsonify(error_response), status_code
    return wrapper

def load_real_data():
    data = {"techniques": [], "cves": []}

    if os.path.exists('data/mitre/enterprise_attack.json'):
        try:
            with open('data/mitre/enterprise_attack.json', 'r') as f:
                attack_data = json.load(f)
                
            for obj in attack_data.get('objects', []):
                if obj.get('type') == 'attack-pattern':
                                      
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

    if os.path.exists('data/api/cisa_kev.csv'):
        try:
            with open('data/api/cisa_kev.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                                                                    
                    cve_id = next((row[key] for key in row.keys() 
                                  if 'cve' in key.lower() and 'id' in key.lower()), None)
                    
                    if cve_id:
                        data["cves"].append(row)
            print(f"Loaded {len(data['cves'])} CVEs from CISA KEV")
        except Exception as e:
            print(f"Error loading CISA KEV data: {e}")
    
    return data

real_data = load_real_data()

@app.route('/', methods=['GET'])
@handle_api_error
def home():
    return jsonify({
        "name": "Echelon Real Data API",
        "data_available": {
            "techniques": len(real_data["techniques"]),
            "cves": len(real_data["cves"])
        }
    })

@app.route('/techniques', methods=['GET'])
@handle_api_error
def get_techniques():
                               
    limit = min(int(request.args.get('limit', 50)), 500)
    offset = int(request.args.get('offset', 0))

    return jsonify({
        "total": len(real_data["techniques"]),
        "techniques": real_data["techniques"][offset:offset+limit]
    })

@app.route('/cves', methods=['GET'])
@handle_api_error
def get_cves():
                               
    limit = min(int(request.args.get('limit', 50)), 500)
    offset = int(request.args.get('offset', 0))

    return jsonify({
        "total": len(real_data["cves"]),
        "cves": real_data["cves"][offset:offset+limit]
    })

if __name__ == '__main__':
                                                                     
    app.run(host='0.0.0.0', port=5050)
