                      
import os
import json
import pickle
import numpy as np
from datetime import datetime
import random

class EnhancedPredictionEngine:

    def __init__(self):
        self.model = None
        self.apt_mappings = []
        self.attack_types = [
            "Spear Phishing", 
            "Malware Injection", 
            "DDoS", 
            "SQL Injection", 
            "Zero-day Exploitation",
            "Supply Chain Attack",
            "Credential Theft",
            "Social Engineering",
            "Watering Hole Attack",
            "Ransomware"
        ]
        self.load_data()
    
    def load_data(self):

        try:
            with open("models/threat_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            print("Loaded base threat model")
        except Exception as e:
            print(f"Error loading base threat model: {e}")

        try:
            with open("data/processed/apt/mappings.json", "r") as f:
                self.apt_mappings = json.load(f)
            print(f"Loaded {len(self.apt_mappings)} APT mappings")
        except Exception as e:
            print(f"Error loading APT mappings: {e}")
    
    def _get_apt_for_cve(self, cve_id=None, base_score=None):
                                                                                
        apt_candidates = []

        for mapping in self.apt_mappings:
                                                  
            pulse_name = mapping.get("pulse_name", "").lower()
            pulse_desc = mapping.get("pulse_description", "").lower()

            if cve_id and (cve_id.lower() in pulse_name or cve_id.lower() in pulse_desc):
                apt_candidates.append({
                    "apt_id": mapping.get("apt_group"),
                    "apt_name": mapping.get("apt_name"),
                    "confidence": mapping.get("confidence"),
                    "reason": f"CVE {cve_id} mentioned in threat intelligence"
                })
                                                                     
            elif mapping.get("confidence", 0) > 0.5:                                          
                apt_candidates.append({
                    "apt_id": mapping.get("apt_group"),
                    "apt_name": mapping.get("apt_name"),
                    "confidence": mapping.get("confidence"),
                    "reason": f"Based on similar threat patterns"
                })

        if apt_candidates:
            apt_candidates.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            return apt_candidates[0]

        if base_score:
            if base_score >= 9.0:                            
                apt_groups = ["apt29", "sandworm", "apt41"]                                                
            elif base_score >= 7.0:                 
                apt_groups = ["apt28", "lazarus", "muddywater"]
            else:                   
                apt_groups = ["apt28", "muddywater"] 

            selected = random.choices(apt_groups, weights=[0.3, 0.4, 0.3], k=1)[0]

            name_map = {
                "apt28": "APT28", 
                "apt29": "APT29", 
                "lazarus": "Lazarus Group",
                "apt41": "APT41",
                "sandworm": "Sandworm Team",
                "muddywater": "MuddyWater"
            }
            
            return {
                "apt_id": selected,
                "apt_name": name_map.get(selected, selected),
                "confidence": 0.6,
                "reason": f"Based on vulnerability severity profile"
            }

        return {
            "apt_id": "unknown",
            "apt_name": "Unknown Actor",
            "confidence": 0.4,
            "reason": "Insufficient data to attribute"
        }
    
    def _predict_attack_type(self, cve_id=None, base_score=None, apt_id=None):

        apt_attack_patterns = {
            "apt28": ["Spear Phishing", "Credential Theft", "Zero-day Exploitation"],
            "apt29": ["Supply Chain Attack", "Spear Phishing", "Malware Injection"],
            "lazarus": ["Watering Hole Attack", "Ransomware", "DDoS"],
            "apt41": ["Supply Chain Attack", "Spear Phishing", "SQL Injection"],
            "sandworm": ["Zero-day Exploitation", "Malware Injection", "DDoS"],
            "muddywater": ["Spear Phishing", "Social Engineering", "Credential Theft"]
        }

        if apt_id and apt_id in apt_attack_patterns:
                                                     
            primary = apt_attack_patterns[apt_id][0]

            others = apt_attack_patterns[apt_id][1:] + random.sample(
                [at for at in self.attack_types if at not in apt_attack_patterns[apt_id]], 
                2
            )
            
            return {
                "primary_attack_type": primary,
                "confidence": 0.75,
                "top_attack_types": [
                    {"type": primary, "probability": 0.75},
                    {"type": others[0], "probability": 0.65},
                    {"type": others[1], "probability": 0.45}
                ]
            }

        if base_score:
            if base_score >= 9.0:                            
                primary = random.choice(["Zero-day Exploitation", "Malware Injection", "Supply Chain Attack"])
                confidence = 0.7
            elif base_score >= 7.0:                 
                primary = random.choice(["Spear Phishing", "SQL Injection", "Credential Theft"])
                confidence = 0.65
            else:                   
                primary = random.choice(["Social Engineering", "Watering Hole Attack", "DDoS"])
                confidence = 0.55

            others = random.sample([at for at in self.attack_types if at != primary], 2)
            
            return {
                "primary_attack_type": primary,
                "confidence": confidence,
                "top_attack_types": [
                    {"type": primary, "probability": confidence},
                    {"type": others[0], "probability": confidence - 0.15},
                    {"type": others[1], "probability": confidence - 0.25}
                ]
            }

        primary = random.choice(self.attack_types)
        others = random.sample([at for at in self.attack_types if at != primary], 2)
        
        return {
            "primary_attack_type": primary,
            "confidence": 0.5,
            "top_attack_types": [
                {"type": primary, "probability": 0.5},
                {"type": others[0], "probability": 0.4},
                {"type": others[1], "probability": 0.3}
            ]
        }
    
    def predict(self, features, cve_id=None):
                                                                                              
        result = {
            "timestamp": datetime.now().isoformat(),
            "input_features": {
                "cve_year": features[0] if len(features) > 0 else None,
                "base_score": features[1] if len(features) > 1 else None,
                "cve_id": cve_id
            }
        }

        if self.model:
            try:
                prediction_proba = self.model.predict_proba([features])[0]
                prediction = int(self.model.predict([features])[0])
                threat_score = float(prediction_proba[1])

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
            except Exception as e:
                print(f"Error making base prediction: {e}")
                result.update({
                    "prediction": 0,
                    "threat_score": 0.5,
                    "threat_level": "MEDIUM",
                    "confidence": 0.5,
                    "error": f"Base model error: {str(e)}"
                })
        else:
            result.update({
                "prediction": 0,
                "threat_score": 0.5,
                "threat_level": "MEDIUM",
                "confidence": 0.5,
                "error": "Base model not loaded"
            })

        apt_attribution = self._get_apt_for_cve(cve_id, features[1] if len(features) > 1 else None)
        result["apt_attribution"] = apt_attribution

        attack_prediction = self._predict_attack_type(
            cve_id, 
            features[1] if len(features) > 1 else None,
            apt_attribution.get("apt_id")
        )
        result["attack_prediction"] = attack_prediction

        try:
            with open("data/processed/geo/threat_locations.json", "r") as f:
                geo_data = json.load(f)

            if geo_data:
                                                   
                recent_points = sorted(geo_data, key=lambda x: x.get("created", ""), reverse=True)[:10]
                result["geographic_data"] = recent_points
        except Exception as e:
            print(f"Error loading geographic data: {e}")
        
        return result

if __name__ == "__main__":
    engine = EnhancedPredictionEngine()

    features = [2023, 8.5, 30]                                    
    prediction = engine.predict(features, cve_id="CVE-2023-20198")
    
    print(json.dumps(prediction, indent=2))
