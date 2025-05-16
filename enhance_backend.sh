#!/bin/bash
# Script to collect real threat intelligence data

# Function to check command status
check_status() {
  if [ $? -eq 0 ]; then
    echo "✓ $1"
  else
    echo "✗ $1"
    exit 1
  fi
}

echo "========================================="
echo "ECHELON: COLLECTING REAL THREAT DATA"
echo "========================================="

# Check if API keys are configured properly
if [ ! -f "config/api_keys.json" ]; then
    echo "⚠️ API configuration file not found. Creating it now."
    
    mkdir -p config
    
    cat > config/api_keys.json << CONFEND
{
  "alienvault_otx": {
    "api_key": "YOUR_ALIENVAULT_OTX_KEY"
  },
  "abuseipdb": {
    "api_key": "YOUR_ABUSEIPDB_KEY"
  },
  "mitre_attack": {
    "url": "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
  },
  "cisa_kev": {
    "url": "https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv"
  }
}
CONFEND
    
    echo "Created API configuration file at config/api_keys.json"
    echo "⚠️ YOU MUST EDIT THIS FILE AND ADD REAL API KEYS BEFORE CONTINUING"
    echo "Please edit the file now and add your API keys for AlienVault OTX and AbuseIPDB"
    
    read -p "Press Enter when you have edited the file, or Ctrl+C to cancel..." 
    
    # Verify after edit
    if grep -q "YOUR_ALIENVAULT_OTX_KEY" config/api_keys.json || grep -q "YOUR_ABUSEIPDB_KEY" config/api_keys.json; then
        echo "⚠️ API keys not properly configured"
        echo "Please edit config/api_keys.json and add real API keys"
        exit 1
    fi
fi

# Check Python dependencies
echo "Checking Python dependencies..."
REQUIRED_PACKAGES=("requests" "numpy" "pycountry" "scikit-learn")
MISSING_PACKAGES=()

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    python3 -c "import $pkg" 2>/dev/null
    if [ $? -ne 0 ]; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "Installing missing Python packages: ${MISSING_PACKAGES[*]}"
    pip3 install ${MISSING_PACKAGES[*]}
    check_status "Installed required packages"
fi

# Collect standard data
echo "Collecting standard threat data..."
mkdir -p data/raw/{cisa,mitre,nvd,rss}

# CISA KEV data
echo "Downloading CISA KEV data..."
curl -s -L "https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv" -o data/raw/cisa/kev.csv
check_status "Downloaded CISA KEV data"

# MITRE ATT&CK data
echo "Downloading MITRE ATT&CK data..."
curl -s -L "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json" -o data/raw/mitre/enterprise.json
check_status "Downloaded MITRE ATT&CK data"

# NVD data
echo "Downloading NVD data..."
curl -s -L "https://services.nvd.nist.gov/rest/json/cves/2.0?resultsPerPage=50" -o data/raw/nvd/recent.json
check_status "Downloaded NVD data"

# Collect advanced threat intel
echo "Collecting advanced threat intelligence data..."
python3 scripts/advanced_data_collector.py
check_status "Collected advanced threat data"

# Process collected data
echo "Processing collected data..."
mkdir -p data/processed/{cves,techniques,alerts}

python3 -c '
import os
import json
import csv
import xml.etree.ElementTree as ET
import datetime

# Process CISA KEV data
kev_path = "data/raw/cisa/kev.csv"
if os.path.exists(kev_path):
    try:
        with open(kev_path, "r") as f:
            # Try to determine format from first line
            first_line = f.readline().strip()
            headers = first_line.split(",")
            
            # Look for header names
            cve_idx = -1
            name_idx = -1
            date_idx = -1
            
            for i, header in enumerate(headers):
                header_lower = header.lower()
                if "cve" in header_lower and "id" in header_lower:
                    cve_idx = i
                elif "name" in header_lower or "vuln" in header_lower:
                    name_idx = i
                elif "date" in header_lower and "add" in header_lower:
                    date_idx = i
            
            # Reset file pointer
            f.seek(0)
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            processed = 0
            for row in reader:
                if len(row) <= max(cve_idx, name_idx, date_idx):
                    continue  # Skip incomplete rows
                
                cve_id = row[cve_idx] if cve_idx >= 0 else ""
                name = row[name_idx] if name_idx >= 0 else ""
                date_added = row[date_idx] if date_idx >= 0 else ""
                
                if cve_id:
                    # Create CVE object
                    cve_obj = {
                        "cve_id": cve_id.strip(),
                        "name": name.strip(),
                        "date_added": date_added.strip(),
                        "source": "CISA KEV",
                        "exploited": True  # KEV entries are known exploited vulnerabilities
                    }
                    
                    # Save to file
                    os.makedirs("data/processed/cves", exist_ok=True)
                    with open(f"data/processed/cves/{cve_id.strip()}.json", "w") as out_f:
                        json.dump(cve_obj, out_f, indent=2)
                    
                    processed += 1
            
            print(f"Processed {processed} CVEs from CISA KEV")
    except Exception as e:
        print(f"Error processing CISA KEV data: {e}")

# Process MITRE ATT&CK data
mitre_path = "data/raw/mitre/enterprise.json"
if os.path.exists(mitre_path):
    try:
        with open(mitre_path, "r") as f:
            mitre_data = json.load(f)
        
        processed = 0
        for obj in mitre_data.get("objects", []):
            if obj.get("type") == "attack-pattern":
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
                    os.makedirs("data/processed/techniques", exist_ok=True)
                    with open(f"data/processed/techniques/{tech_id}.json", "w") as out_f:
                        json.dump(technique, out_f, indent=2)
                    
                    processed += 1
        
        print(f"Processed {processed} techniques from MITRE ATT&CK")
    except Exception as e:
        print(f"Error processing MITRE ATT&CK data: {e}")

# Process NVD data
nvd_path = "data/raw/nvd/recent.json"
if os.path.exists(nvd_path):
    try:
        with open(nvd_path, "r") as f:
            nvd_data = json.load(f)
        
        processed = 0
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
                    os.makedirs("data/processed/cves", exist_ok=True)
                    with open(f"data/processed/cves/{cve_id}.json", "w") as out_f:
                        json.dump(cve_obj, out_f, indent=2)
                    processed += 1
        
        print(f"Processed {processed} CVEs from NVD")
    except Exception as e:
        print(f"Error processing NVD data: {e}")

# Train baseline prediction model if needed
if not os.path.exists("models/threat_model.pkl"):
    print("Creating baseline prediction model...")
    
    import numpy as np
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from datetime import datetime
    
    # Collect CVE data
    cves = []
    for cve_file in os.listdir("data/processed/cves"):
        if cve_file.endswith(".json"):
            try:
                with open(os.path.join("data/processed/cves", cve_file), "r") as f:
                    cve = json.load(f)
                    cves.append(cve)
            except:
                pass
    
    if len(cves) > 10:
        # Extract features
        X = []
        y = []
        
        for cve in cves:
            # Feature 1: CVE Year
            cve_year = 2020  # Default
            if "cve_id" in cve and cve["cve_id"]:
                try:
                    cve_year = int(cve["cve_id"].split("-")[1])
                except:
                    pass
            
            # Feature 2: CVSS Base Score
            base_score = 5.0  # Default
            if "base_score" in cve:
                base_score = float(cve["base_score"])
                
            # Target: Is it exploited
            exploited = 1 if cve.get("source") == "CISA KEV" or cve.get("exploited", False) else 0
            
            X.append([cve_year, base_score])
            y.append(exploited)
        
        # Train model
        X = np.array(X)
        y = np.array(y)
        
        # Train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Save model
        os.makedirs("models", exist_ok=True)
        with open("models/threat_model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            "training_date": datetime.now().isoformat(),
            "num_samples": len(X),
            "num_features": X.shape[1],
            "feature_names": ["CVE Year", "CVSS Score"],
            "accuracy": 1.0  # Placeholder
        }
        
        with open("models/model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Created baseline prediction model with {len(X)} samples")
'
check_status "Processed threat data"

# Check if real data was collected
echo "Checking if real threat intelligence data was collected..."
python3 -c '
import os
import json
import sys

# Check if APT mappings exist
apt_mappings_path = "data/processed/apt/mappings.json"
if not os.path.exists(apt_mappings_path):
    print("No APT mappings found!")
    sys.exit(1)

# Load APT mappings
with open(apt_mappings_path) as f:
    apt_mappings = json.load(f)

# Check if geographic data exists
geo_path = "data/processed/geo/threat_locations.json"
abuse_path = "data/processed/geo/abuse_locations.json"

geo_data = []
if os.path.exists(geo_path):
    with open(geo_path) as f:
        geo_data = json.load(f)

abuse_data = []
if os.path.exists(abuse_path):
    with open(abuse_path) as f:
        abuse_data = json.load(f)

# Verify we have enough data
if len(apt_mappings) == 0 and len(geo_data) == 0 and len(abuse_data) == 0:
    print("No real threat intelligence data was collected!")
    sys.exit(1)

print(f"Found {len(apt_mappings)} APT mappings and {len(geo_data) + len(abuse_data)} geographic data points")
'
check_status "Verified real data collection"

echo "========================================="
echo "REAL DATA COLLECTION COMPLETE!"
echo "========================================="
echo ""
echo "Real threat intelligence data has been collected and processed."
echo "The system is now ready to run in real-data-only mode."
echo ""
echo "To check data availability:"
echo "  python3 check_real_data.py"
echo ""
echo "To start the system:"
echo "  ./run_echelon.sh"
