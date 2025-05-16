#!/usr/bin/env python3
import os
import json
import csv
import requests
import time
import sys
from datetime import datetime
import ipaddress
import pycountry
import xml.etree.ElementTree as ET

# Load API configuration
def load_api_config():
    try:
        with open('config/api_keys.json', 'r') as f:
            return json.load(f)
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
        if not self.api_key or self.api_key == "YOUR_ALIENVAULT_OTX_KEY":
            print("AlienVault OTX API key not configured")
            return []
            
        url = f"{self.base_url}/pulses/subscribed"
        params = {
            "limit": limit
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            print(f"Error fetching AlienVault OTX pulses: {e}")
            return []
    
    def get_indicators(self, pulse_id):
        """Get indicators for a specific pulse"""
        url = f"{self.base_url}/pulses/{pulse_id}/indicators"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
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
        if not self.api_key or self.api_key == "YOUR_ABUSEIPDB_KEY":
            print("AbuseIPDB API key not configured")
            return {}
            
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
                    "apt_group": apt,
                    "confidence_score": min(score / 10, 1.0),  # Normalize to 0-1
                    "reasons": reasons
                })
        
        # Sort by confidence score
        matches.sort(key=lambda x: x["confidence_score"], reverse=True)
        return matches

# Geographic data processor using real data
class GeoProcessor:
    def __init__(self):
        self.country_codes = {country.alpha_2: country.name for country in pycountry.countries}
    
    def ip_to_country(self, ip):
        """Get country for an IP using AbuseIPDB or other source (simplified)"""
        # In a real implementation, this would use MaxMind GeoIP or similar
        # For demo purposes, we'll simulate with a fixed mapping
        try:
            # Check if it's a valid IP
            ipaddress.ip_address(ip)
            
            # In a real implementation, use a proper IP geolocation service
            # For now, return a placeholder
            return None
        except:
            return None
    
    def process_indicators(self, indicators):
        """Process indicators to extract geographic information"""
        geo_data = []
        
        for indicator in indicators:
            if indicator.get("type") == "IPv4" or indicator.get("type") == "IPv6":
                ip = indicator.get("indicator")
                country = self.ip_to_country(ip)
                
                if country:
                    geo_data.append({
                        "ip": ip,
                        "country": country,
                        "type": indicator.get("type"),
                        "created": indicator.get("created")
                    })
        
        return geo_data

# Main data collection and processing
def main():
    print("Starting advanced data collection...")
    
    # Create data directories
    os.makedirs("data/raw/otx", exist_ok=True)
    os.makedirs("data/raw/abuseipdb", exist_ok=True)
    os.makedirs("data/processed/geo", exist_ok=True)
    os.makedirs("data/processed/apt", exist_ok=True)
    
    # Load API configuration
    config = load_api_config()
    
    # Initialize API clients
    otx = AlienVaultOTX(config["alienvault_otx"]["api_key"])
    abuse_ip = AbuseIPDB(config["abuseipdb"]["api_key"])
    apt_mapper = APTMapper()
    geo_processor = GeoProcessor()
    
    # Collect data from AlienVault OTX
    print("Collecting data from AlienVault OTX...")
    pulses = otx.get_pulses(limit=30)
    
    # Save raw pulses
    with open("data/raw/otx/pulses.json", "w") as f:
        json.dump(pulses, f, indent=2)
    
    # Process pulses for APT attribution
    apt_mappings = []
    geographic_data = []
    
    print(f"Processing {len(pulses)} threat intelligence pulses...")
    for pulse in pulses:
        # Map pulse to APT groups
        apt_matches = apt_mapper.map_pulse_to_apt_groups(pulse)
        
        if apt_matches:
            timestamp = datetime.now().isoformat()
            pulse_id = pulse.get("id")
            
            # Get indicators for the pulse
            indicators = otx.get_indicators(pulse_id)
            
            # Extract geographic data
            geo_data = geo_processor.process_indicators(indicators)
            
            # Add to geographic dataset
            geographic_data.extend(geo_data)
            
            # Prepare mapping record
            for match in apt_matches:
                mapping = {
                    "pulse_id": pulse_id,
                    "pulse_name": pulse.get("name"),
                    "pulse_description": pulse.get("description"),
                    "pulse_created": pulse.get("created"),
                    "apt_group": match["apt_group"]["id"],
                    "apt_name": match["apt_group"]["name"],
                    "confidence": match["confidence_score"],
                    "reasons": match["reasons"],
                    "processed_at": timestamp
                }
                
                apt_mappings.append(mapping)
    
    # Save APT mappings
    with open("data/processed/apt/mappings.json", "w") as f:
        json.dump(apt_mappings, f, indent=2)
    
    # Save geographic data
    with open("data/processed/geo/threat_locations.json", "w") as f:
        json.dump(geographic_data, f, indent=2)
    
    # Collect data from AbuseIPDB
    print("Collecting data from AbuseIPDB...")
    blacklist = abuse_ip.get_blacklist(limit=100)
    
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
    
    print("Advanced data collection complete!")

if __name__ == "__main__":
    main()
