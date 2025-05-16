#!/usr/bin/env python3
import os
import json
import pickle
import numpy as np
from datetime import datetime

class EnhancedPredictionEngine:
    """Enhanced prediction engine that strictly uses real data with no fallbacks or synthetic data"""
    
    def __init__(self):
        self.model = None
        self.apt_mappings = []
        self.load_data()
    
    def load_data(self):
        """Load necessary prediction data"""
        # Load base threat model
        try:
            with open("models/threat_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            print("Loaded base threat model")
        except Exception as e:
            print(f"Error loading base threat model: {e}")
            self.model = None
        
        # Load APT mappings from processed data
        try:
            with open("data/processed/apt/mappings.json", "r") as f:
                self.apt_mappings = json.load(f)
            print(f"Loaded {len(self.apt_mappings)} APT mappings")
        except Exception as e:
            print(f"Error loading APT mappings: {e}")
            self.apt_mappings = []
    
    def _get_apt_for_cve(self, cve_id=None, base_score=None):
        """Determine APT group for a CVE based ONLY on real intel data - no synthetic matching"""
        apt_candidates = []
        
        # Use real mappings ONLY
        for mapping in self.apt_mappings:
            # Check if this mapping mentions the specific CVE
            pulse_name = mapping.get("pulse_name", "").lower()
            pulse_desc = mapping.get("pulse_description", "").lower()
            
            # If we're looking for a specific CVE and it's mentioned
            if cve_id and (cve_id.lower() in pulse_name or cve_id.lower() in pulse_desc):
                apt_candidates.append({
                    "apt_id": mapping.get("apt_group"),
                    "apt_name": mapping.get("apt_name"),
                    "confidence": mapping.get("confidence"),
                    "reason": f"CVE {cve_id} mentioned in threat intelligence"
                })
        
        # If we have candidates with real evidence, return the highest confidence match
        if apt_candidates:
            apt_candidates.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            return apt_candidates[0]
        
        # If no direct mapping exists, return None instead of guessing
        return None
    
    def _get_attack_types_for_apt(self, apt_id):
        """Get real attack types for an APT based on intel data - no synthetic data"""
        attack_types = []
        
        # Search through all the mappings for this APT group
        for mapping in self.apt_mappings:
            if mapping.get("apt_group") == apt_id:
                # Look for attack types in the pulse data
                pulse_desc = mapping.get("pulse_description", "").lower()
                
                # Only extract attack types that are explicitly mentioned
                techniques = []
                if "phishing" in pulse_desc:
                    techniques.append("Spear Phishing")
                if "supply chain" in pulse_desc:
                    techniques.append("Supply Chain Attack")
                if "credential" in pulse_desc or "password" in pulse_desc:
                    techniques.append("Credential Theft")
                if "ddos" in pulse_desc or "denial of service" in pulse_desc:
                    techniques.append("DDoS")
                if "zero-day" in pulse_desc or "0day" in pulse_desc:
                    techniques.append("Zero-day Exploitation")
                if "sql" in pulse_desc or "injection" in pulse_desc:
                    techniques.append("SQL Injection")
                if "social engineering" in pulse_desc:
                    techniques.append("Social Engineering")
                if "watering hole" in pulse_desc:
                    techniques.append("Watering Hole Attack")
                if "ransomware" in pulse_desc:
                    techniques.append("Ransomware")
                if "malware" in pulse_desc:
                    techniques.append("Malware Injection")
                
                # Add any found techniques to our list (avoid duplicates)
                for technique in techniques:
                    if technique not in [t.get("type") for t in attack_types]:
                        attack_types.append({
                            "type": technique,
                            "source": mapping.get("pulse_id"),
                            "confidence": mapping.get("confidence", 0.5)
                        })
        
        return attack_types
    
    def predict(self, features, cve_id=None):
        """Make prediction using ONLY real data, no synthetic fallbacks"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "input_features": {
                "cve_year": features[0] if len(features) > 0 else None,
                "base_score": features[1] if len(features) > 1 else None,
                "cve_id": cve_id
            },
            "using_real_data_only": True
        }
        
        # Base threat prediction (using existing model)
        if self.model:
            try:
                # Only predict if we have the minimum features needed
                if len(features) >= 2:
                    prediction_proba = self.model.predict_proba([features])[0]
                    prediction = int(self.model.predict([features])[0])
                    threat_score = float(prediction_proba[1])
                    
                    # Determine threat level
                    threat_level = "LOW"
                    if threat_score > 0.7:
                        threat_level = "HIGH"
                    elif threat_score > 0.4:
                        threat_level = "MEDIUM"
                    
                    result.update({
                        "prediction": prediction,
                        "threat_score": round(threat_score, 4),
                        "threat_level": threat_level,
                        "confidence": round(abs(prediction_proba[1] - prediction_proba[0]), 4)
                    })
                else:
                    result.update({
                        "error": "Insufficient features for prediction",
                        "required_features": ["cve_year", "base_score"]
                    })
            except Exception as e:
                print(f"Error making base prediction: {e}")
                result.update({
                    "error": f"Base model error: {str(e)}"
                })
        else:
            result.update({
                "error": "Base model not loaded"
            })
        
        # APT attribution - only include if we have real data
        apt_attribution = self._get_apt_for_cve(cve_id, features[1] if len(features) > 1 else None)
        if apt_attribution:
            result["apt_attribution"] = apt_attribution
            
            # Add attack types, but only if we have the APT attribution
            attack_types = self._get_attack_types_for_apt(apt_attribution.get("apt_id"))
            if attack_types:
                # Sort by confidence
                attack_types.sort(key=lambda x: x.get("confidence", 0), reverse=True)
                
                # Add to result
                if len(attack_types) > 0:
                    result["attack_prediction"] = {
                        "primary_attack_type": attack_types[0]["type"],
                        "confidence": attack_types[0]["confidence"],
                        "top_attack_types": attack_types[:3] if len(attack_types) >= 3 else attack_types
                    }
        
        # Get geographic data if available - only real data
        try:
            geo_data = []
            if os.path.exists("data/processed/geo/threat_locations.json"):
                with open("data/processed/geo/threat_locations.json", "r") as f:
                    geo_data = json.load(f)
            
            # Add geographical data if available
            if geo_data:
                result["geographic_data"] = geo_data
        except Exception as e:
            print(f"Error loading geographic data: {e}")
        
        return result

# For testing
if __name__ == "__main__":
    engine = EnhancedPredictionEngine()
    
    # Test with a sample CVE
    features = [2023, 8.5, 30]  # Year, CVSS, days since published
    prediction = engine.predict(features, cve_id="CVE-2023-20198")
    
    print(json.dumps(prediction, indent=2))
