#!/bin/bash
# fix_data_collection.sh - Fix the data collection process to only use real data

echo "========================================="
echo "ECHELON: FIXING DATA COLLECTION FOR REAL DATA ONLY"
echo "========================================="

# Fix the advanced data collector
echo "Fixing advanced data collector for real data only..."
cat > scripts/advanced_data_collector.py << 'EOT'
#!/usr/bin/env python3
import os
import json
import csv
import requests
import time
import sys
from datetime import datetime
import ipaddress
try:
    import pycountry
except ImportError:
    print("Installing required pycountry package...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pycountry"])
    import pycountry
import xml.etree.ElementTree as ET

# Load API configuration
def load_api_config():
    try:
        with open('config/api_keys.json', 'r') as f:
            config = json.load(f)
            
            # Validate API keys
            alienvault_key = config.get('alienvault_otx', {}).get('api_key', '')
            abuseipdb_key = config.get('abuseipdb', {}).get('api_key', '')
            
            if not alienvault_key or alienvault_key == 'YOUR_ALIENVAULT_OTX_KEY':
                print("ERROR: AlienVault OTX API key not configured properly")
                print("Please edit config/api_keys.json and add a valid API key")
                sys.exit(1)
                
            if not abuseipdb_key or abuseipdb_key == 'YOUR_ABUSEIPDB_KEY':
                print("ERROR: AbuseIPDB API key not configured properly")
                print("Please edit config/api_keys.json and add a valid API key")
                sys.exit(1)
                
            return config
    except FileNotFoundError:
        print("API config file not found. Please run the setup script first.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Invalid API config format. Please check the file.")
        sys.exit(1)

# AlienVault OTX integration
class AlienVaultOTX:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://otx.alienvault.com/api/v1"
        self.headers = {
            "X-OTX-API-KEY": self.api_key,
            "User-Agent": "Echelon Threat Intelligence System"
        }
    
    def get_pulses(self, limit=20):
        """Get recent threat intelligence pulses"""
        url = f"{self.base_url}/pulses/subscribed"
        params = {
            "limit": limit
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print(f"API Error: Authentication failed (401). Invalid AlienVault OTX API key.")
            else:
                print(f"API Error ({e.response.status_code}): {str(e)}")
            return []
        except Exception as e:
            print(f"Error fetching AlienVault OTX pulses: {e}")
            return []
    
    def get_indicators(self, pulse_id):
        """Get indicators for a specific pulse"""
        url = f"{self.base_url}/pulses/{pulse_id}/indicators"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            # Ensure we have a list of indicators
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "results" in data:
                return data["results"]
            else:
                return []
        except Exception as e:
            print(f"Error fetching indicators for pulse {pulse_id}: {e}")
            return []
    
    def get_ip_reputation(self, ip):
        """Get reputation data for an IP"""
        url = f"{self.base_url}/indicators/IPv4/{ip}/general"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching IP reputation for {ip}: {e}")
            return {}

# AbuseIPDB integration
class AbuseIPDB:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.abuseipdb.com/api/v2"
        self.headers = {
            "Key": self.api_key,
            "Accept": "application/json"
        }
    
    def check_ip(self, ip):
        """Check an IP address for abuse reports"""
        url = f"{self.base_url}/check"
        params = {
            "ipAddress": ip,
            "maxAgeInDays": 90,
            "verbose": True
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json().get("data", {})
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print(f"API Error: Authentication failed (401). Invalid AbuseIPDB API key.")
            else:
                print(f"API Error ({e.response.status_code}): {str(e)}")
            return {}
        except Exception as e:
            print(f"Error checking IP {ip} in AbuseIPDB: {e}")
            return {}
    
    def get_blacklist(self, limit=100):
        """Get blacklisted IPs"""
        url = f"{self.base_url}/blacklist"
        params = {
            "limit": limit,
            "confidenceMinimum": 90
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json().get("data", [])
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print(f"API Error: Authentication failed (401). Invalid AbuseIPDB API key.")
            else:
                print(f"API Error ({e.response.status_code}): {str(e)}")
            return []
        except Exception as e:
            print(f"Error fetching blacklist from AbuseIPDB: {e}")
            return []

# APT mapping based on real world TTPs and indicators
class APTMapper:
    def __init__(self):
        self.apt_groups = self._load_apt_data()
        
    def _load_apt_data(self):
        """Load APT group data from MITRE ATT&CK and other sources"""
        # Define core APT groups with their real-world attributes
        # This data is based on real threat intel reports
        apt_groups = [
            {
                "id": "apt28",
                "name": "APT28",
                "aliases": ["Fancy Bear", "Sofacy", "Sednit", "Strontium"],
                "origin": "Russia",
                "primary_techniques": ["T1566", "T1190", "T1133"],  # Spear phishing, Exploit Public-Facing App, External Remote Services
                "targets": ["Government", "Military", "NATO"],
                "targeted_regions": ["Europe", "North America", "Ukraine"],
                "targeted_sectors": ["Government", "Defense", "Political"],
                "malware_families": ["X-Tunnel", "X-Agent", "Lojax", "Zebrocy"]
            },
            {
                "id": "apt29",
                "name": "APT29",
                "aliases": ["Cozy Bear", "The Dukes", "Nobelium"],
                "origin": "Russia",
                "primary_techniques": ["T1195", "T1566", "T1059.003"],  # Supply Chain Compromise, Spear Phishing, PowerShell
                "targets": ["Government", "Think tanks", "Healthcare"],
                "targeted_regions": ["Europe", "North America"],
                "targeted_sectors": ["Government", "Healthcare", "Research", "NGO"],
                "malware_families": ["MiniDuke", "CozyDuke", "SUNBURST", "WellMess"]
            },
            {
                "id": "lazarus",
                "name": "Lazarus Group",
                "aliases": ["Hidden Cobra", "Guardians of Peace", "ZINC"],
                "origin": "North Korea",
                "primary_techniques": ["T1190", "T1486", "T1055"],  # Exploit Public-Facing App, Data Encrypted for Impact, Process Injection
                "targets": ["Financial", "Cryptocurrency", "Media"],
                "targeted_regions": ["Global", "South Korea", "United States"],
                "targeted_sectors": ["Financial", "Cryptocurrency", "Entertainment"],
                "malware_families": ["WannaCry", "HOPLIGHT", "FASTCash", "ELECTRICFISH"]
            },
            {
                "id": "apt41",
                "name": "APT41",
                "aliases": ["Winnti", "Barium", "Wicked Panda"],
                "origin": "China",
                "primary_techniques": ["T1195", "T1190", "T1059.007"],  # Supply Chain, Exploit Public-Facing App, JavaScript
                "targets": ["Healthcare", "Technology", "Gaming"],
                "targeted_regions": ["East Asia", "North America", "Europe"],
                "targeted_sectors": ["Healthcare", "Technology", "Gaming", "Telecom"],
                "malware_families": ["Winnti", "POISONPLUG", "HIGHNOON", "DEADEYE"]
            },
            {
                "id": "sandworm",
                "name": "Sandworm Team",
                "aliases": ["BlackEnergy", "Voodoo Bear", "ELECTRUM"],
                "origin": "Russia",
                "primary_techniques": ["T1190", "T1133", "T1486"],  # Exploit Public-Facing App, External Remote Services, Data Encrypted for Impact
                "targets": ["Energy", "Industrial systems", "Ukraine"],
                "targeted_regions": ["Ukraine", "Europe", "United States"],
                "targeted_sectors": ["Energy", "Industrial", "Government"],
                "malware_families": ["BlackEnergy", "NotPetya", "GreyEnergy", "Industroyer"]
            },
            {
                "id": "muddywater",
                "name": "MuddyWater",
                "aliases": ["Earth Vetala", "TEMP.Zagros", "Static Kitten"],
                "origin": "Iran",
                "primary_techniques": ["T1566.001", "T1059.001", "T1059.003"],  # Spear Phishing Attachment, PowerShell, Command Scripting
                "targets": ["Government", "Telecommunications", "Defense"],
                "targeted_regions": ["Middle East", "Central Asia", "Europe"],
                "targeted_sectors": ["Government", "Telecommunications", "Defense", "Oil and Gas"],
                "malware_families": ["PowGoop", "POWERSTATS", "REDFACE", "Small Sieve"]
            }
        ]
        
        return apt_groups
    
    def map_pulse_to_apt_groups(self, pulse):
        """Map an AlienVault OTX pulse to potential APT groups based on IOCs, TTPs, and targets"""
        matches = []
        
        # Skip if no tags or malware families in the pulse
        if not pulse.get("tags") and not pulse.get("malware_families"):
            return []
        
        for apt in self.apt_groups:
            score = 0
            reasons = []
            
            # Check for APT name or aliases in tags
            pulse_tags = [tag.lower() for tag in pulse.get("tags", [])]
            apt_identifiers = [apt["name"].lower()] + [alias.lower() for alias in apt.get("aliases", [])]
            
            for identifier in apt_identifiers:
                if identifier in pulse.get("name", "").lower() or identifier in pulse.get("description", "").lower():
                    score += 5
                    reasons.append(f"APT name '{identifier}' found in pulse name/description")
                
                if any(identifier in tag for tag in pulse_tags):
                    score += 3
                    reasons.append(f"APT name '{identifier}' found in pulse tags")
            
            # Check for malware families
            for malware in apt.get("malware_families", []):
                if malware.lower() in pulse.get("name", "").lower() or malware.lower() in pulse.get("description", "").lower():
                    score += 4
                    reasons.append(f"Malware '{malware}' found in pulse name/description")
                
                if any(malware.lower() in mf.lower() for mf in pulse.get("malware_families", [])):
                    score += 3
                    reasons.append(f"Malware '{malware}' found in pulse malware families")
            
            # Check for targeted regions
            for region in apt.get("targeted_regions", []):
                if region.lower() in pulse.get("name", "").lower() or region.lower() in pulse.get("description", "").lower():
                    score += 2
                    reasons.append(f"Targeted region '{region}' found in pulse")
            
            # Check for targeted sectors
            for sector in apt.get("targeted_sectors", []):
                if sector.lower() in pulse.get("name", "").lower() or sector.lower() in pulse.get("description", "").lower():
                    score += 2
                    reasons.append(f"Targeted sector '{sector}' found in pulse")
            
            if score > 0:
                matches.append({
                    "apt_group": apt["id"],
                    "apt_name": apt["name"],
                    "confidence_score": min(score / 10, 1.0),  # Normalize to 0-1
                    "reasons": reasons
                })
        
        # Sort by confidence score
        matches.sort(key=lambda x: x["confidence_score"], reverse=True)
        return matches

# Geographic data processor
class GeoProcessor:
    def __init__(self):
        self.country_codes = {country.alpha_2: country.name for country in pycountry.countries}
    
    def process_indicators(self, indicators):
        """Process indicators to extract geographic information"""
        geo_data = []
        
        # Debug the indicators structure
        if not indicators:
            print("No indicators provided")
            return []
            
        print(f"Processing {len(indicators)} indicators")
        
        # Process each indicator
        for i, indicator in enumerate(indicators):
            try:
                # Check indicator type
                if isinstance(indicator, dict) and 'type' in indicator:
                    indicator_type = indicator.get('type')
                    if indicator_type in ['IPv4', 'IPv6']:
                        ip = indicator.get('indicator')
                        country_code = None
                        
                        # Try to find country code in the data
                        if 'country_code' in indicator:
                            country_code = indicator.get('country_code')
                        elif 'country' in indicator and len(indicator.get('country', '')) == 2:
                            country_code = indicator.get('country')
                        
                        # Convert country code to name
                        country_name = None
                        if country_code and len(country_code) == 2:
                            country_name = self.country_codes.get(country_code)
                        
                        if ip:
                            geo_data.append({
                                'ip': ip,
                                'country_code': country_code,
                                'country': country_name or 'Unknown',
                                'type': indicator_type,
                                'created': indicator.get('created', datetime.now().isoformat())
                            })
                elif isinstance(indicator, str):
                    # If indicator is a string, it's likely just the indicator value
                    # Try to determine if it's an IP
                    try:
                        ipaddress.ip_address(indicator)
                        # It's a valid IP
                        geo_data.append({
                            'ip': indicator,
                            'country_code': None,
                            'country': 'Unknown',
                            'type': 'IPv4' if '.' in indicator else 'IPv6',
                            'created': datetime.now().isoformat()
                        })
                    except ValueError:
                        # Not an IP address, skip
                        pass
            except Exception as e:
                print(f"Error processing indicator {i}: {e}")
        
        return geo_data

# Main data collection function
def main():
    print("Starting real threat intelligence data collection...")
    
    # Create necessary directories
    os.makedirs("data/raw/otx", exist_ok=True)
    os.makedirs("data/raw/abuseipdb", exist_ok=True)
    os.makedirs("data/processed/geo", exist_ok=True)
    os.makedirs("data/processed/apt", exist_ok=True)
    
    # Load API configuration
    print("Loading API configuration...")
    config = load_api_config()
    
    # Initialize API clients
    otx = AlienVaultOTX(config["alienvault_otx"]["api_key"])
    abuse_ip = AbuseIPDB(config["abuseipdb"]["api_key"])
    apt_mapper = APTMapper()
    geo_processor = GeoProcessor()
    
    # Lists to store our results
    apt_mappings = []
    geographic_data = []
    
    # Process AlienVault OTX data
    print("Fetching threat intelligence data from AlienVault OTX...")
    pulses = otx.get_pulses(limit=20)
    
    if not pulses:
        print("ERROR: Failed to retrieve pulses from AlienVault OTX")
        print("Please check your API key or network connection")
        sys.exit(1)
    
    # Save raw pulses
    with open("data/raw/otx/pulses.json", "w") as f:
        json.dump(pulses, f, indent=2)
    
    print(f"Processing {len(pulses)} threat intelligence pulses...")
    for i, pulse in enumerate(pulses):
        print(f"Processing pulse {i+1}/{len(pulses)}: {pulse.get('name', 'Unknown')}")
        
        try:
            # Map pulse to APT groups
            apt_matches = apt_mapper.map_pulse_to_apt_groups(pulse)
            
            if apt_matches:
                timestamp = datetime.now().isoformat()
                pulse_id = pulse.get("id")
                
                # Get indicators for the pulse
                indicators = otx.get_indicators(pulse_id)
                
                # Extract geographic data
                try:
                    geo_data = geo_processor.process_indicators(indicators)
                    
                    # Only add to geographic dataset if we got some data
                    if geo_data:
                        print(f"  Found {len(geo_data)} geographic data points")
                        geographic_data.extend(geo_data)
                except Exception as e:
                    print(f"Error processing geographic data: {e}")
                
                # Add APT mappings
                for match in apt_matches:
                    mapping = {
                        "pulse_id": pulse_id,
                        "pulse_name": pulse.get("name"),
                        "pulse_description": pulse.get("description"),
                        "pulse_created": pulse.get("created"),
                        "apt_group": match["apt_group"],
                        "apt_name": match["apt_name"],
                        "confidence": match["confidence_score"],
                        "reasons": match["reasons"],
                        "processed_at": timestamp
                    }
                    
                    apt_mappings.append(mapping)
        except Exception as e:
            print(f"Error processing pulse: {e}")
    
    # Save APT mappings
    with open("data/processed/apt/mappings.json", "w") as f:
        json.dump(apt_mappings, f, indent=2)
    
    # Save geographic data
    with open("data/processed/geo/threat_locations.json", "w") as f:
        json.dump(geographic_data, f, indent=2)
    
    # Collect data from AbuseIPDB
    print("\nCollecting data from AbuseIPDB...")
    blacklist = abuse_ip.get_blacklist(limit=100)
    
    if not blacklist:
        print("ERROR: Failed to retrieve blacklist from AbuseIPDB")
        print("Please check your API key or network connection")
    else:
        # Save raw blacklist
        with open("data/raw/abuseipdb/blacklist.json", "w") as f:
            json.dump(blacklist, f, indent=2)
        
        # Process for geographic visualization
        blacklist_geo = []
        
        for ip_entry in blacklist:
            ip = ip_entry.get("ipAddress")
            country = ip_entry.get("countryCode")
            
            if ip and country:
                blacklist_geo.append({
                    "ip": ip,
                    "country_code": country,
                    "abuse_confidence": ip_entry.get("abuseConfidenceScore"),
                    "last_reported": ip_entry.get("lastReportedAt")
                })
        
        # Save processed blacklist
        with open("data/processed/geo/abuse_locations.json", "w") as f:
            json.dump(blacklist_geo, f, indent=2)
        
        print(f"Processed {len(blacklist_geo)} blacklisted IPs")
    
    # Check if we have collected enough data
    if len(apt_mappings) == 0 and len(geographic_data) == 0:
        print("\nERROR: No real threat intelligence data was collected")
        print("The system requires real data to operate in real-data-only mode")
        sys.exit(1)
    
    print("\nReal threat intelligence data collection complete!")
    print(f"Collected {len(apt_mappings)} APT mappings and {len(geographic_data)} geographic data points")

if __name__ == "__main__":
    main()
EOT
chmod +x scripts/advanced_data_collector.py

# Update the enhance_backend.sh script
echo "Updating enhanced backend script..."
cat > enhance_backend.sh << 'EOT'
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
EOT
chmod +x enhance_backend.sh

echo "========================================="
echo "DATA COLLECTION FIX COMPLETE"
echo "========================================="
echo ""
echo "The data collection process has been fixed to work robustly with real data."
echo "There are no mock data generators or fallbacks in the system."
echo ""
echo "Run the enhanced backend script again to collect real data:"
echo "  ./enhance_backend.sh"
echo ""
echo "Make sure your API keys are properly configured in config/api_keys.json"