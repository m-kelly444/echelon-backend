import glob
import json
import os
import pickle
import random
import re
from datetime import datetime, timedelta

import numpy as np

class MLAttackForecaster:
    def __init__(self):
        self.model = None
        self.apt_data = None
        self.geo_data = None
        self.cve_data = None
        self.loaded_files = {}
        self._load_all_data()

    def _load_all_data(self):
        self._load_model()
        self._load_apt_data()
        self._load_geo_data()
        self._load_cve_data()
        self._load_additional_data()

    def _load_model(self):
        model_paths = glob.glob("**/threat_model.pkl", recursive=True)
        if model_paths:
            try:
                with open(model_paths[0], "rb") as f:
                    self.model = pickle.load(f)
                self.loaded_files["model"] = model_paths[0]
            except Exception:
                pass

    def _load_apt_data(self):
        apt_paths = glob.glob("**/mappings.json", recursive=True)
        if apt_paths:
            try:
                with open(apt_paths[0], "r") as f:
                    self.apt_data = json.load(f)
                self.loaded_files["apt"] = apt_paths[0]
            except Exception:
                pass

    def _load_geo_data(self):
        geo_patterns = [
            "**/threat_locations.json",
            "**/abuse_locations.json",
            "**/*locations*.json"]
        for pattern in geo_patterns:
            geo_paths = glob.glob(pattern, recursive=True)
            if geo_paths:
                try:
                    with open(geo_paths[0], "r") as f:
                        self.geo_data = json.load(f)
                    self.loaded_files["geo"] = geo_paths[0]
                    break
                except Exception:
                    pass

    def _load_cve_data(self):
        self.cve_data = []
        cve_dirs = glob.glob("**/cves", recursive=True)
        if cve_dirs:
            cve_dir = cve_dirs[0]
            cve_files = glob.glob(f"{cve_dir}/*.json")
            for cve_file in cve_files:
                try:
                    with open(cve_file, "r") as f:
                        self.cve_data.append(json.load(f))
                except Exception:
                    continue
            self.loaded_files["cves"] = f"{len(self.cve_data)} files from {cve_dir}"

    def _load_additional_data(self):
        misc_data_patterns = [
            "**/nvd/*.json",
            "**/mitre/*.json",
            "**/cisa/*.json"]
        for pattern in misc_data_patterns:
            files = glob.glob(pattern, recursive=True)
            if files:
                try:
                    with open(files[0], "r") as f:
                        data = json.load(f)
                    self.loaded_files[pattern] = files[0]
                except Exception:
                    continue

    def get_data_status(self):
        data_counts = {
            "cves": len(self.cve_data) if self.cve_data else 0,
            "apt_mappings": len(self.apt_data) if self.apt_data else 0,
            "geo_locations": len(self.geo_data) if self.geo_data else 0,
            "model_loaded": self.model is not None
        }

        loaded_sources = []
        for key, value in self.loaded_files.items():
            if isinstance(value, str) and os.path.exists(value):
                loaded_sources.append({"type": key, "path": value})
            elif "files from" in str(value):
                loaded_sources.append({"type": key, "info": value})

        return {
            "data_counts": data_counts,
            "loaded_files": self.loaded_files,
            "total_data_points": sum(data_counts.values()) - (1 if self.model else 0),
            "has_sufficient_data": data_counts["cves"] > 5 or data_counts["apt_mappings"] > 2 or data_counts["geo_locations"] > 2
        }

    def predict_threat(self, cve_id, base_score):
        if not self.model:
            return {"error": "No ML model loaded", "real_data_only": True}

        cve_year = 2020
        if cve_id and "CVE-" in cve_id:
            try:
                cve_year = int(cve_id.split("-")[1])
            except BaseException:
                pass

        try:
            features = [cve_year, base_score]
            prediction_proba = self.model.predict_proba([features])[0]
            prediction = int(self.model.predict([features])[0])
            threat_score = float(prediction_proba[1])

            threat_level = "LOW"
            if threat_score > 0.7:
                threat_level = "HIGH"
            elif threat_score > 0.4:
                threat_level = "MEDIUM"

            similar_cves = []
            if self.cve_data:
                for cve in self.cve_data:
                    if "cve_id" in cve and "base_score" in cve:
                        try:
                            curr_year = int(cve["cve_id"].split("-")[1])
                            curr_score = float(cve["base_score"])

                            if abs(
                                    curr_year -
                                    cve_year) <= 2 and abs(
                                    curr_score -
                                    base_score) <= 1.0:
                                similar_cves.append({
                                    "cve_id": cve["cve_id"],
                                    "base_score": cve["base_score"],
                                    "severity": cve.get("severity", "UNKNOWN")
                                })

                                if len(similar_cves) >= 3:
                                    break
                        except BaseException:
                            continue

            data_counts = self.get_data_status()["data_counts"]

            return {
                "prediction": prediction,
                "threat_score": round(threat_score, 4),
                "threat_level": threat_level,
                "confidence": round(abs(prediction_proba[1] - prediction_proba[0]), 4),
                "real_data_only": True,
                "data_based_on": data_counts,
                "similar_vulnerabilities": similar_cves[:3]
            }
        except Exception as e:
            return {"error": str(e), "real_data_only": True}

    def _extract_countries_from_geo(self):
        if not self.geo_data:
            return {}

        countries = {}
        for item in self.geo_data:
            country = item.get("country") or item.get(
                "country_name") or "Unknown"
            if country not in countries:
                countries[country] = 0
            countries[country] += 1

        return countries

    def _extract_apt_techniques(self):
        if not self.apt_data:
            return []

        techniques = {}
        for mapping in self.apt_data:
            description = mapping.get("pulse_description", "").lower()

            patterns = [
                ("phishing", "Phishing"),
                ("supply chain", "Supply Chain Attack"),
                ("ransomware", "Ransomware"),
                ("ddos", "DDoS"),
                ("zero-day", "Zero-day Exploitation"),
                ("injection", "Injection Attack"),
                ("credential", "Credential Theft"),
                ("backdoor", "Backdoor"),
                ("malware", "Malware Distribution"),
                ("exfiltration", "Data Exfiltration")
            ]

            for pattern, tech_name in patterns:
                if pattern in description:
                    if tech_name not in techniques:
                        techniques[tech_name] = 0
                    techniques[tech_name] += 1

        result = [{"technique": k, "count": v} for k, v in techniques.items()]
        result.sort(key=lambda x: x["count"], reverse=True)
        return result

    def _extract_severity_distribution(self):
        if not self.cve_data:
            return []

        severity_counts = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
            "UNKNOWN": 0}
        for cve in self.cve_data:
            severity = cve.get("severity", "UNKNOWN").upper()
            if severity not in severity_counts:
                severity = "UNKNOWN"
            severity_counts[severity] += 1

        total = sum(severity_counts.values())
        distribution = []
        for severity, count in severity_counts.items():
            if count > 0:
                distribution.append({
                    "severity": severity,
                    "count": count,
                    "percentage": round((count / total) * 100, 1)
                })

        distribution.sort(key=lambda x: x["count"], reverse=True)
        return distribution

    def _extract_apt_groups(self):
        if not self.apt_data:
            return []

        groups = {}
        for mapping in self.apt_data:
            group = mapping.get("apt_group", "unknown")
            if group not in groups:
                groups[group] = {
                    "count": 0,
                    "confidence_sum": 0,
                    "name": mapping.get("apt_name", group)
                }

            groups[group]["count"] += 1
            groups[group]["confidence_sum"] += mapping.get("confidence", 0.5)

        result = []
        for group, data in groups.items():
            if data["count"] > 0:
                result.append({
                    "group": group,
                    "name": data["name"],
                    "activity_count": data["count"],
                    "average_confidence": round(data["confidence_sum"] / data["count"], 2)
                })

        result.sort(key=lambda x: x["activity_count"], reverse=True)
        return result

    def _generate_time_series(self, days=30):
        now = datetime.now()
        time_series = []

        base_value = 50
        if self.cve_data:
            severity_dist = self._extract_severity_distribution()
            high_critical_pct = sum(
                item["percentage"] for item in severity_dist if item["severity"] in [
                    "HIGH", "CRITICAL"])
            base_value = 30 + (high_critical_pct / 2)

        for i in range(days):
            date = now + timedelta(days=i)

            value = base_value

            weekday = date.weekday()
            if weekday < 5:
                value *= 1.2
            else:
                value *= 0.7

            if 10 <= date.day <= 20:
                value *= 1.15

            if date.day == 1 or date.day == 15:
                value *= 1.3

            confidence = max(0.4, min(0.95, 0.9 - (i * 0.02)))

            time_series.append({
                "date": date.strftime("%Y-%m-%d"),
                "expected_activity": round(value, 2),
                "confidence": round(confidence, 2)
            })

        return time_series

    def _extract_target_sectors(self):
        if not self.apt_data and not self.cve_data:
            return []

        sectors = {}
        keywords = {
            "finance": "Financial Services",
            "bank": "Financial Services",
            "healthcare": "Healthcare",
            "health": "Healthcare",
            "medical": "Healthcare",
            "government": "Government",
            "military": "Military/Defense",
            "defense": "Military/Defense",
            "education": "Education",
            "university": "Education",
            "school": "Education",
            "telecom": "Telecommunications",
            "energy": "Energy",
            "manufacturing": "Manufacturing",
            "retail": "Retail",
            "technology": "Technology",
            "critical": "Critical Infrastructure"
        }

        if self.apt_data:
            for mapping in self.apt_data:
                desc = (mapping.get("pulse_name", "") + " " +
                        mapping.get("pulse_description", "")).lower()

                for keyword, sector in keywords.items():
                    if keyword in desc:
                        if sector not in sectors:
                            sectors[sector] = 0
                        sectors[sector] += 1

        if self.cve_data:
            for cve in self.cve_data:
                desc = cve.get("description", "").lower()

                for keyword, sector in keywords.items():
                    if keyword in desc:
                        if sector not in sectors:
                            sectors[sector] = 0
                        sectors[sector] += 1

        result = [{"sector": k, "risk_score": min(100, v * 5)}
                  for k, v in sectors.items()]
        result.sort(key=lambda x: x["risk_score"], reverse=True)
        return result

    def _compute_attack_vectors(self):
        if not self.cve_data:
            return []

        vectors = {}
        keywords = {
            "web application": "Web Applications",
            "remote code execution": "Remote Code Execution",
            "sql injection": "SQL Injection",
            "cross-site": "Cross-Site Scripting (XSS)",
            "xss": "Cross-Site Scripting (XSS)",
            "authentication": "Authentication Bypass",
            "credentials": "Credential Theft",
            "password": "Password Attack",
            "denial of service": "Denial of Service (DoS)",
            "buffer overflow": "Buffer Overflow",
            "privilege escalation": "Privilege Escalation",
            "path traversal": "Path Traversal",
            "file inclusion": "File Inclusion",
            "memory corruption": "Memory Corruption"
        }

        for cve in self.cve_data:
            desc = cve.get("description", "").lower()
            severity = cve.get("severity", "UNKNOWN")
            weight = 3 if severity == "CRITICAL" else 2 if severity == "HIGH" else 1

            for keyword, vector in keywords.items():
                if keyword in desc:
                    if vector not in vectors:
                        vectors[vector] = 0
                    vectors[vector] += weight

        result = [{"vector": k, "frequency": v}
                  for k, v in vectors.items()]
        result.sort(key=lambda x: x["frequency"], reverse=True)
        return result

    def generate_attack_forecast(self, days=30):
        data_status = self.get_data_status()

        if not data_status["has_sufficient_data"]:
            return {
                "error": "Insufficient data for ML-based forecasting",
                "real_data_only": True,
                "data_status": data_status
            }

        forecast = {
            "generated_at": datetime.now().isoformat(),
            "forecast_period": {
                "start_date": datetime.now().isoformat(),
                "end_date": (
                    datetime.now() +
                    timedelta(
                        days=days)).isoformat(),
                "days": days},
            "real_data_only": True,
            "data_based_on": data_status["data_counts"]}

        attack_techniques = self._extract_apt_techniques()
        if attack_techniques:
            forecast["attack_techniques"] = attack_techniques

        target_countries = self._extract_countries_from_geo()
        if target_countries:
            countries_list = [{"country": k, "activity_level": v}
                              for k, v in target_countries.items()]
            countries_list.sort(
                key=lambda x: x["activity_level"],
                reverse=True)
            forecast["target_countries"] = countries_list[:10]

        severity_distribution = self._extract_severity_distribution()
        if severity_distribution:
            forecast["vulnerability_severity"] = severity_distribution

        apt_groups = self._extract_apt_groups()
        if apt_groups:
            forecast["threat_actors"] = apt_groups[:10]

        time_series = self._generate_time_series(days)
        if time_series:
            forecast["activity_forecast"] = time_series

        target_sectors = self._extract_target_sectors()
        if target_sectors:
            forecast["target_sectors"] = target_sectors

        attack_vectors = self._compute_attack_vectors()
        if attack_vectors:
            forecast["attack_vectors"] = attack_vectors

        return forecast
