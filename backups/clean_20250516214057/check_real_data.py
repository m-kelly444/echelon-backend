#!/usr/bin/env python3
import os
import json
from datetime import datetime

def check_data_source(path, name, required=True):
    """Check if a data source exists and has data"""
    if not os.path.exists(path):
        if required:
            print(f"❌ MISSING REQUIRED DATA: {name} ({path})")
            return False
        else:
            print(f"⚠️ Optional data not found: {name} ({path})")
            return False
    
    size = os.path.getsize(path)
    if size == 0:
        if required:
            print(f"❌ EMPTY REQUIRED DATA: {name} ({path})")
            return False
        else:
            print(f"⚠️ Optional data file is empty: {name} ({path})")
            return False
    
    print(f"✓ {name} data available: {size} bytes")
    return True

def check_api_keys():
    """Check if API keys are configured"""
    if not os.path.exists('config/api_keys.json'):
        print("❌ API keys configuration file not found")
        return False
    
    try:
        with open('config/api_keys.json', 'r') as f:
            config = json.load(f)
        
        alienvault_key = config.get('alienvault_otx', {}).get('api_key', '')
        abuseipdb_key = config.get('abuseipdb', {}).get('api_key', '')
        
        missing_keys = []
        
        if not alienvault_key or alienvault_key == 'YOUR_ALIENVAULT_OTX_KEY':
            missing_keys.append("AlienVault OTX")
        
        if not abuseipdb_key or abuseipdb_key == 'YOUR_ABUSEIPDB_KEY':
            missing_keys.append("AbuseIPDB")
        
        if missing_keys:
            print(f"❌ Missing API keys: {', '.join(missing_keys)}")
            print("   Please configure these in config/api_keys.json")
            return False
        
        print("✓ API keys configured")
        return True
    
    except Exception as e:
        print(f"❌ Error checking API keys: {e}")
        return False

def main():
    print("========================================")
    print(" ECHELON REAL DATA REQUIREMENTS CHECK")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("========================================")
    
    # Track overall status
    all_required_available = True
    
    # Check API keys
    print("\nChecking API keys:")
    if not check_api_keys():
        all_required_available = False
    
    # Check required data sources
    print("\nChecking required data sources:")
    
    # Raw data
    if not check_data_source('data/raw/cisa/kev.csv', 'CISA KEV', required=True):
        all_required_available = False
    
    if not check_data_source('data/raw/mitre/enterprise.json', 'MITRE ATT&CK', required=True):
        all_required_available = False
    
    if not check_data_source('data/raw/nvd/recent.json', 'NVD', required=True):
        all_required_available = False
    
    # Processed data
    if not check_data_source('data/processed/apt/mappings.json', 'APT Mappings', required=True):
        all_required_available = False
    
    if not check_data_source('data/processed/geo/threat_locations.json', 'Geographic Threat Data', required=True):
        all_required_available = False
    
    # Check if model is available
    if not check_data_source('models/threat_model.pkl', 'Threat Model', required=True):
        all_required_available = False
    
    # Result
    print("\n========================================")
    if all_required_available:
        print("✅ ALL REQUIRED REAL DATA IS AVAILABLE")
        print("   The system can run in real-data-only mode")
    else:
        print("❌ SOME REQUIRED REAL DATA IS MISSING")
        print("   Please run enhance_backend.sh to collect the required data")
    
    return 0 if all_required_available else 1

if __name__ == "__main__":
    main()
