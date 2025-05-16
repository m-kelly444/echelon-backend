#!/usr/bin/env python3
import os
import json
import csv
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import glob

app = Flask(__name__)
CORS(app)

# Load all real data - no fallbacks, no mocks
def load_all_data():
    data = {
        "cves": [],
        "techniques": [],
        "alerts": []
    }
    
    # Load CISA KEV data (CSV)
    try:
        if os.path.exists('data/api/cisa_kev.csv'):
            with open('data/api/cisa_kev.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Only add if we have the necessary data
                    if 'cveID' in row and row['cveID']:
                        data["cves"].append({
                            "id": row.get('cveID', ''),
                            "name": row.get('vulnerabilityName', ''),
                            "date_added": row.get('dateAdded', ''),
                            "source": "CISA KEV",
                            "required_action": row.get('requiredAction', ''),
                            "due_date": row.get('dueDate', '')
                        })
            print(f"Loaded {len(data['cves'])} CVEs from CISA KEV")
    except Exception as e:
        print(f"Error loading CISA KEV data: {e}")

    # Load NVD data
    try:
        if os.path.exists('data/api/nvd_api.json'):
            with open('data/api/nvd_api.json', 'r') as f:
                nvd_data = json.load(f)
                if 'vulnerabilities' in nvd_data:
                    for vuln in nvd_data['vulnerabilities']:
                        cve = vuln.get('cve', {})
                        if 'id' in cve:
                            # Extract what we can from NVD
                            data["cves"].append({
                                "id": cve.get('id', ''),
                                "description": cve.get('descriptions', [{}])[0].get('value', '') if cve.get('descriptions') else '',
                                "published": cve.get('published', ''),
                                "source": "NVD"
                            })
            print(f"Added {len(nvd_data.get('vulnerabilities', []))} CVEs from NVD")
    except Exception as e:
        print(f"Error loading NVD data: {e}")

    # Load MITRE ATT&CK data
    try:
        mitre_files = ['enterprise_attack.json', 'mobile_attack.json', 'ics_attack.json']
        for file_name in mitre_files:
            file_path = os.path.join('data/mitre', file_name)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    mitre_data = json.load(f)
                    if 'objects' in mitre_data:
                        for obj in mitre_data['objects']:
                            if obj.get('type') == 'attack-pattern':
                                technique = {
                                    "id": next((ref.get('external_id', '') for ref in obj.get('external_references', []) 
                                               if ref.get('source_name') == 'mitre-attack'), ''),
                                    "name": obj.get('name', ''),
                                    "description": obj.get('description', ''),
                                    "source": f"MITRE {file_name.split('_')[0].upper()} ATT&CK"
                                }
                                if technique["id"]:  # Only add if we have an ID
                                    data["techniques"].append(technique)
                print(f"Added {sum(1 for t in data['techniques'] if t['source'].startswith('MITRE ' + file_name.split('_')[0].upper()))} techniques from {file_name}")
    except Exception as e:
        print(f"Error loading MITRE data: {e}")

    # Load RSS feed data
    try:
        for rss_file in glob.glob('data/rss/*.xml'):
            try:
                tree = ET.parse(rss_file)
                root = tree.getroot()
                
                # RSS format
                items = root.findall('.//item')
                feed_name = os.path.basename(rss_file).replace('.xml', '')
                
                for item in items:
                    title = item.find('title')
                    link = item.find('link')
                    pub_date = item.find('pubDate')
                    
                    if title is not None and title.text:
                        alert = {
                            "title": title.text,
                            "link": link.text if link is not None and link.text else "",
                            "date": pub_date.text if pub_date is not None and pub_date.text else "",
                            "source": feed_name
                        }
                        data["alerts"].append(alert)
                
                print(f"Added {len([a for a in data['alerts'] if a['source'] == feed_name])} alerts from {feed_name}")
            except Exception as e:
                print(f"Error parsing RSS file {rss_file}: {e}")
    except Exception as e:
        print(f"Error loading RSS data: {e}")
    
    return data

# Load all data at startup
all_data = load_all_data()

# API endpoints that use only real data
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "name": "Echelon Real-Data API",
        "version": "1.0.0",
        "data_counts": {
            "cves": len(all_data["cves"]),
            "techniques": len(all_data["techniques"]),
            "alerts": len(all_data["alerts"])
        },
        "endpoints": [
            {"path": "/cves", "method": "GET", "description": "Get real vulnerability data"},
            {"path": "/techniques", "method": "GET", "description": "Get real ATT&CK techniques"},
            {"path": "/alerts", "method": "GET", "description": "Get real security alerts"}
        ]
    })

@app.route('/cves', methods=['GET'])
def get_cves():
    limit = min(int(request.args.get('limit', 50)), 500)
    offset = int(request.args.get('offset', 0))
    search = request.args.get('search', '')
    
    results = all_data["cves"]
    
    # Apply search if provided
    if search:
        results = [cve for cve in results if 
                   search.lower() in cve.get('id', '').lower() or 
                   search.lower() in cve.get('name', '').lower() or
                   search.lower() in cve.get('description', '').lower()]
    
    # Apply pagination
    paginated = results[offset:offset+limit]
    
    return jsonify({
        "total": len(results),
        "count": len(paginated),
        "offset": offset,
        "limit": limit,
        "cves": paginated
    })

@app.route('/techniques', methods=['GET'])
def get_techniques():
    limit = min(int(request.args.get('limit', 50)), 500)
    offset = int(request.args.get('offset', 0))
    
    # Apply pagination
    paginated = all_data["techniques"][offset:offset+limit]
    
    return jsonify({
        "total": len(all_data["techniques"]),
        "count": len(paginated),
        "offset": offset,
        "limit": limit,
        "techniques": paginated
    })

@app.route('/alerts', methods=['GET'])
def get_alerts():
    limit = min(int(request.args.get('limit', 20)), 100)
    offset = int(request.args.get('offset', 0))
    source = request.args.get('source', '')
    
    results = all_data["alerts"]
    
    # Filter by source if provided
    if source:
        results = [alert for alert in results if source.lower() in alert.get('source', '').lower()]
    
    # Apply pagination
    paginated = results[offset:offset+limit]
    
    return jsonify({
        "total": len(results),
        "count": len(paginated),
        "offset": offset,
        "limit": limit,
        "alerts": paginated
    })

@app.route('/cve/<cve_id>', methods=['GET'])
def get_cve(cve_id):
    # Find the specific CVE
    for cve in all_data["cves"]:
        if cve.get('id', '').upper() == cve_id.upper():
            return jsonify(cve)
    
    return jsonify({"error": "CVE not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
