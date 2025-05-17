                      
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

def load_api_config():
    try:
        with open('config/api_keys.json', 'r') as f:
            config = json.load(f)

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

class AlienVaultOTX:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://otx.alienvault.com/api/v1"
        self.headers = {
            "X-OTX-API-KEY": self.api_key,
            "User-Agent": "Echelon Threat Intelligence System"
        }
    
    def get_pulses(self, limit=20):
                                                   
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
                                                 
        url = f"{self.base_url}/pulses/{pulse_id}/indicators"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

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
                                           
        url = f"{self.base_url}/indicators/IPv4/{ip}/general"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching IP reputation for {ip}: {e}")
            return {}

class AbuseIPDB:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.abuseipdb.com/api/v2"
        self.headers = {
            "Key": self.api_key,
            "Accept": "application/json"
        }
    
    def check_ip(self, ip):
                                                   
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

class APTMapper:
    def __init__(self):
        self.apt_groups = self._load_apt_data()
        
    def _load_apt_data(self):

        apt_groups = [
            {
                "id": "apt28",
                "name": "APT28",
                "aliases": ["Fancy Bear", "Sofacy", "Sednit", "Strontium"],
                "origin": "Russia",
                "primary_techniques": ["T1566", "T1190", "T1133"],                                                                       
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
                "primary_techniques": ["T1195", "T1566", "T1059.003"],                                                       
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
                "primary_techniques": ["T1190", "T1486", "T1055"],                                                                           
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
                "primary_techniques": ["T1195", "T1190", "T1059.007"],                                                       
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
                "primary_techniques": ["T1190", "T1133", "T1486"],                                                                                  
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
                "primary_techniques": ["T1566.001", "T1059.001", "T1059.003"],                                                            
                "targets": ["Government", "Telecommunications", "Defense"],
                "targeted_regions": ["Middle East", "Central Asia", "Europe"],
                "targeted_sectors": ["Government", "Telecommunications", "Defense", "Oil and Gas"],
                "malware_families": ["PowGoop", "POWERSTATS", "REDFACE", "Small Sieve"]
            }
        ]
        
        return apt_groups
    
    def map_pulse_to_apt_groups(self, pulse):
                                                                                                  
        matches = []

        if not pulse.get("tags") and not pulse.get("malware_families"):
            return []
        
        for apt in self.apt_groups:
            score = 0
            reasons = []

            pulse_tags = [tag.lower() for tag in pulse.get("tags", [])]
            apt_identifiers = [apt["name"].lower()] + [alias.lower() for alias in apt.get("aliases", [])]
            
            for identifier in apt_identifiers:
                if identifier in pulse.get("name", "").lower() or identifier in pulse.get("description", "").lower():
                    score += 5
                    reasons.append(f"APT name '{identifier}' found in pulse name/description")
                
                if any(identifier in tag for tag in pulse_tags):
                    score += 3
                    reasons.append(f"APT name '{identifier}' found in pulse tags")

            for malware in apt.get("malware_families", []):
                if malware.lower() in pulse.get("name", "").lower() or malware.lower() in pulse.get("description", "").lower():
                    score += 4
                    reasons.append(f"Malware '{malware}' found in pulse name/description")
                
                if any(malware.lower() in mf.lower() for mf in pulse.get("malware_families", [])):
                    score += 3
                    reasons.append(f"Malware '{malware}' found in pulse malware families")

            for region in apt.get("targeted_regions", []):
                if region.lower() in pulse.get("name", "").lower() or region.lower() in pulse.get("description", "").lower():
                    score += 2
                    reasons.append(f"Targeted region '{region}' found in pulse")

            for sector in apt.get("targeted_sectors", []):
                if sector.lower() in pulse.get("name", "").lower() or sector.lower() in pulse.get("description", "").lower():
                    score += 2
                    reasons.append(f"Targeted sector '{sector}' found in pulse")
            
            if score > 0:
                matches.append({
                    "apt_group": apt["id"],
                    "apt_name": apt["name"],
                    "confidence_score": min(score / 10, 1.0),                    
                    "reasons": reasons
                })

        matches.sort(key=lambda x: x["confidence_score"], reverse=True)
        return matches

class GeoProcessor:
    def __init__(self):
        self.country_codes = {country.alpha_2: country.name for country in pycountry.countries}
    
    def process_indicators(self, indicators):
                                                                  
        geo_data = []

        if not indicators:
            print("No indicators provided")
            return []
            
        print(f"Processing {len(indicators)} indicators")

        for i, indicator in enumerate(indicators):
            try:
                                      
                if isinstance(indicator, dict) and 'type' in indicator:
                    indicator_type = indicator.get('type')
                    if indicator_type in ['IPv4', 'IPv6']:
                        ip = indicator.get('indicator')
                        country_code = None

                        if 'country_code' in indicator:
                            country_code = indicator.get('country_code')
                        elif 'country' in indicator and len(indicator.get('country', '')) == 2:
                            country_code = indicator.get('country')

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

                    try:
                        ipaddress.ip_address(indicator)
                                         
                        geo_data.append({
                            'ip': indicator,
                            'country_code': None,
                            'country': 'Unknown',
                            'type': 'IPv4' if '.' in indicator else 'IPv6',
                            'created': datetime.now().isoformat()
                        })
                    except ValueError:
                                                 
                        pass
            except Exception as e:
                print(f"Error processing indicator {i}: {e}")
        
        return geo_data

def main():
    print("Starting real threat intelligence data collection...")

    os.makedirs("data/raw/otx", exist_ok=True)
    os.makedirs("data/raw/abuseipdb", exist_ok=True)
    os.makedirs("data/processed/geo", exist_ok=True)
    os.makedirs("data/processed/apt", exist_ok=True)

    print("Loading API configuration...")
    config = load_api_config()

    otx = AlienVaultOTX(config["alienvault_otx"]["api_key"])
    abuse_ip = AbuseIPDB(config["abuseipdb"]["api_key"])
    apt_mapper = APTMapper()
    geo_processor = GeoProcessor()

    apt_mappings = []
    geographic_data = []

    print("Fetching threat intelligence data from AlienVault OTX...")
    pulses = otx.get_pulses(limit=20)
    
    if not pulses:
        print("ERROR: Failed to retrieve pulses from AlienVault OTX")
        print("Please check your API key or network connection")
        sys.exit(1)

    with open("data/raw/otx/pulses.json", "w") as f:
        json.dump(pulses, f, indent=2)
    
    print(f"Processing {len(pulses)} threat intelligence pulses...")
    for i, pulse in enumerate(pulses):
        print(f"Processing pulse {i+1}/{len(pulses)}: {pulse.get('name', 'Unknown')}")
        
        try:
                                     
            apt_matches = apt_mapper.map_pulse_to_apt_groups(pulse)
            
            if apt_matches:
                timestamp = datetime.now().isoformat()
                pulse_id = pulse.get("id")

                indicators = otx.get_indicators(pulse_id)

                try:
                    geo_data = geo_processor.process_indicators(indicators)

                    if geo_data:
                        print(f"  Found {len(geo_data)} geographic data points")
                        geographic_data.extend(geo_data)
                except Exception as e:
                    print(f"Error processing geographic data: {e}")

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

    with open("data/processed/apt/mappings.json", "w") as f:
        json.dump(apt_mappings, f, indent=2)

    with open("data/processed/geo/threat_locations.json", "w") as f:
        json.dump(geographic_data, f, indent=2)

    print("\nCollecting data from AbuseIPDB...")
    blacklist = abuse_ip.get_blacklist(limit=100)
    
    if not blacklist:
        print("ERROR: Failed to retrieve blacklist from AbuseIPDB")
        print("Please check your API key or network connection")
    else:
                            
        with open("data/raw/abuseipdb/blacklist.json", "w") as f:
            json.dump(blacklist, f, indent=2)

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

        with open("data/processed/geo/abuse_locations.json", "w") as f:
            json.dump(blacklist_geo, f, indent=2)
        
        print(f"Processed {len(blacklist_geo)} blacklisted IPs")

    if len(apt_mappings) == 0 and len(geographic_data) == 0:
        print("\nERROR: No real threat intelligence data was collected")
        print("The system requires real data to operate in real-data-only mode")
        sys.exit(1)
    
    print("\nReal threat intelligence data collection complete!")
    print(f"Collected {len(apt_mappings)} APT mappings and {len(geographic_data)} geographic data points")

if __name__ == "__main__":
    main()
