#!/usr/bin/env python3
import os
import json
import sys
from datetime import datetime

def validate_data_source(path, source_name):
    """Validate that a data source exists and has content"""
    if not os.path.exists(path):
        print(f"✗ {source_name} data not found at {path}")
        return False
    
    # Check file size
    size = os.path.getsize(path)
    if size == 0:
        print(f"✗ {source_name} data is empty")
        return False
    
    print(f"✓ {source_name} data available ({size} bytes)")
    return True

def validate_json_data(path, source_name, required_keys=None):
    """Validate that JSON data exists, has content, and contains required keys"""
    if not validate_data_source(path, source_name):
        return False
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            if len(data) == 0:
                print(f"✗ {source_name} data is an empty list")
                return False
            print(f"✓ {source_name} data contains {len(data)} records")
            
            if required_keys and len(data) > 0:
                # Check first item for required keys
                first_item = data[0]
                missing_keys = [key for key in required_keys if key not in first_item]
                if missing_keys:
                    print(f"✗ {source_name} data missing required keys: {', '.join(missing_keys)}")
                    return False
        elif isinstance(data, dict):
            if len(data) == 0:
                print(f"✗ {source_name} data is an empty dictionary")
                return False
            print(f"✓ {source_name} data contains {len(data)} keys")
            
            if required_keys:
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    print(f"✗ {source_name} data missing required keys: {', '.join(missing_keys)}")
                    return False
        
        return True
    except json.JSONDecodeError:
        print(f"✗ {source_name} data is not valid JSON")
        return False
    except Exception as e:
        print(f"✗ Error validating {source_name} data: {str(e)}")
        return False

def main():
    print("========================================")
    print("ECHELON REAL DATA VALIDATION")
    print(f"Run at: {datetime.now().isoformat()}")
    print("========================================")
    
    all_valid = True
    
    # Check for API keys - needed for real data
    config_valid = validate_json_data('config/api_keys.json', 'API Keys Configuration')
    if config_valid:
        try:
            with open('config/api_keys.json', 'r') as f:
                config = json.load(f)
            
            alienvault_key = config.get('alienvault_otx', {}).get('api_key', '')
            abuseipdb_key = config.get('abuseipdb', {}).get('api_key', '')
            
            if alienvault_key == '' or alienvault_key == 'YOUR_ALIENVAULT_OTX_KEY':
                print("✗ AlienVault OTX API key not configured")
                all_valid = False
            else:
                print("✓ AlienVault OTX API key configured")
            
            if abuseipdb_key == '' or abuseipdb_key == 'YOUR_ABUSEIPDB_KEY':
                print("✗ AbuseIPDB API key not configured")
                all_valid = False
            else:
                print("✓ AbuseIPDB API key configured")
        except Exception as e:
            print(f"✗ Error checking API keys: {str(e)}")
            all_valid = False
    else:
        all_valid = False
    
    # Check for raw data sources
    print("\nChecking raw data sources:")
    if not validate_data_source('data/raw/cisa/kev.csv', 'CISA KEV'):
        all_valid = False
    
    if not validate_json_data('data/raw/mitre/enterprise.json', 'MITRE ATT&CK'):
        all_valid = False
    
    if not validate_json_data('data/raw/nvd/recent.json', 'NVD'):
        all_valid = False
    
    if not validate_json_data('data/raw/otx/pulses.json', 'AlienVault OTX Pulses'):
        all_valid = False
    
    if not validate_json_data('data/raw/abuseipdb/blacklist.json', 'AbuseIPDB Blacklist'):
        all_valid = False
    
    # Check for processed data
    print("\nChecking processed data:")
    if not validate_json_data('data/processed/apt/mappings.json', 'APT Mappings'):
        all_valid = False
    
    if not validate_json_data('data/processed/geo/threat_locations.json', 'Geographic Threat Data'):
        all_valid = False
    
    # Check for model
    print("\nChecking prediction model:")
    if not validate_data_source('models/threat_model.pkl', 'Prediction Model'):
        all_valid = False
    
    if not validate_json_data('models/model_metadata.json', 'Model Metadata'):
        all_valid = False
    
    # Summary
    print("\n========================================")
    if all_valid:
        print("✅ ALL REAL DATA SOURCES VALIDATED SUCCESSFULLY")
        print("The system is ready to run with real data only.")
        return 0
    else:
        print("❌ SOME REAL DATA SOURCES ARE MISSING OR INVALID")
        print("Please run the data collection scripts with valid API keys to ensure all real data is available.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
