#!/bin/bash
# Script to process collected data into a format suitable for analysis

source config.sh

# Create processed data directories
mkdir -p data/processed/{cves,techniques,alerts}

echo "Starting data processing at $(date)"

# Process CISA KEV data
if [ -s "data/raw/cisa/kev.csv" ]; then
  echo "Processing CISA KEV data..."
  
  # Extract CVE IDs and other relevant fields
  tail -n +2 data/raw/cisa/kev.csv | while IFS=, read -r cveID vulnName product vendorProject dateAdded shortDescription requiredAction dueDate
  do
    # Clean up fields and escape quotes
    cveID=$(echo "$cveID" | tr -d '"')
    vulnName=$(echo "$vulnName" | tr -d '"')
    dateAdded=$(echo "$dateAdded" | tr -d '"')
    
    # Output processed data in JSON format
    echo "{\"cve_id\":\"$cveID\",\"name\":\"$vulnName\",\"date_added\":\"$dateAdded\",\"source\":\"CISA KEV\",\"exploited\":true}" > "data/processed/cves/$cveID.json"
  done
  
  echo "  ✓ CISA KEV data processed"
else
  echo "  ✗ No CISA KEV data to process"
fi

# Process MITRE ATT&CK data using Python (for JSON parsing)
if [ -s "data/raw/mitre/enterprise.json" ]; then
  echo "Processing MITRE ATT&CK data..."
  
  # Use Python for better JSON handling
  python3 -c '
import json
import os

# Load MITRE ATT&CK data
with open("data/raw/mitre/enterprise.json", "r") as f:
    mitre_data = json.load(f)

# Process attack patterns
for obj in mitre_data.get("objects", []):
    if obj.get("type") == "attack-pattern":
        # Get technique ID
        tech_id = ""
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                tech_id = ref.get("external_id", "")
                break
        
        if tech_id:
            # Create technique object
            technique = {
                "id": tech_id,
                "name": obj.get("name", ""),
                "description": obj.get("description", ""),
                "source": "MITRE ATT&CK"
            }
            
            # Save to file
            with open(f"data/processed/techniques/{tech_id}.json", "w") as out_file:
                json.dump(technique, out_file)
                
print(f"Processed {len([f for f in os.listdir(\"data/processed/techniques\") if f.endswith(\".json\")])} ATT&CK techniques")
'
  
  echo "  ✓ MITRE ATT&CK data processed"
else
  echo "  ✗ No MITRE ATT&CK data to process"
fi

# Process NVD data using Python (for JSON parsing)
if [ -s "data/raw/nvd/recent.json" ]; then
  echo "Processing NVD data..."
  
  # Use Python for better JSON handling
  python3 -c '
import json
import os

# Load NVD data
with open("data/raw/nvd/recent.json", "r") as f:
    nvd_data = json.load(f)

# Process vulnerabilities
count = 0
for vuln in nvd_data.get("vulnerabilities", []):
    cve = vuln.get("cve", {})
    cve_id = cve.get("id", "")
    
    if cve_id:
        # Extract description
        description = ""
        if cve.get("descriptions"):
            for desc in cve.get("descriptions", []):
                if desc.get("lang") == "en":
                    description = desc.get("value", "")
                    break
        
        # Extract CVSS data
        metrics = cve.get("metrics", {})
        cvss_v3 = None
        severity = "UNKNOWN"
        base_score = 0
        
        # Try CVSS v3.1
        if metrics.get("cvssMetricV31"):
            cvss_v3 = metrics.get("cvssMetricV31")[0].get("cvssData", {})
            severity = cvss_v3.get("baseSeverity", "UNKNOWN")
            base_score = float(cvss_v3.get("baseScore", 0))
        # Try CVSS v3.0
        elif metrics.get("cvssMetricV30"):
            cvss_v3 = metrics.get("cvssMetricV30")[0].get("cvssData", {})
            severity = cvss_v3.get("baseSeverity", "UNKNOWN")
            base_score = float(cvss_v3.get("baseScore", 0))
        
        # Create CVE object
        cve_obj = {
            "cve_id": cve_id,
            "description": description,
            "published": cve.get("published", ""),
            "last_modified": cve.get("lastModified", ""),
            "severity": severity,
            "base_score": base_score,
            "source": "NVD"
        }
        
        # Skip if already exists (from CISA KEV, which takes precedence)
        if not os.path.exists(f"data/processed/cves/{cve_id}.json"):
            with open(f"data/processed/cves/{cve_id}.json", "w") as out_file:
                json.dump(cve_obj, out_file)
                count += 1
                
print(f"Processed {count} CVEs from NVD")
'
  
  echo "  ✓ NVD data processed"
else
  echo "  ✗ No NVD data to process"
fi

# Process RSS feeds
echo "Processing RSS feeds..."
python3 -c '
import os
import xml.etree.ElementTree as ET
import json
import hashlib
from datetime import datetime

def parse_rss(file_path, source_name):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        items = []
        
        # Try RSS format
        for item in root.findall(".//item"):
            title_elem = item.find("title")
            link_elem = item.find("link")
            date_elem = item.find("pubDate")
            
            if title_elem is not None and title_elem.text:
                alert = {
                    "title": title_elem.text,
                    "link": link_elem.text if link_elem is not None and link_elem.text else "",
                    "date": date_elem.text if date_elem is not None and date_elem.text else "",
                    "source": source_name,
                    "processed_at": datetime.now().isoformat()
                }
                
                # Generate ID
                alert_id = hashlib.md5(alert["title"].encode()).hexdigest()
                
                # Save to file
                with open(f"data/processed/alerts/{alert_id}.json", "w") as out_file:
                    json.dump(alert, out_file)
                    items.append(alert_id)
        
        return len(items)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return 0

# Process all RSS files
total = 0
for rss_file in os.listdir("data/raw/rss"):
    if rss_file.endswith(".xml"):
        file_path = os.path.join("data/raw/rss", rss_file)
        source_name = os.path.splitext(rss_file)[0]
        count = parse_rss(file_path, source_name)
        total += count
        print(f"  Processed {count} alerts from {source_name}")

print(f"Total processed alerts: {total}")
'

echo "Data processing completed at $(date)"
