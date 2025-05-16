#!/bin/bash
# Script to extract features and train the prediction model

source config.sh

# Create model directory
mkdir -p models

echo "Starting feature extraction and model training at $(date)"

# Use Python for feature extraction and model training
python3 -c '
import os
import json
import pickle
import numpy as np
import re
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Process CVEs
print("Extracting features from CVEs...")
cves = []
cve_dir = "data/processed/cves"

for cve_file in os.listdir(cve_dir):
    if cve_file.endswith(".json"):
        try:
            with open(os.path.join(cve_dir, cve_file), "r") as f:
                cve = json.load(f)
                cves.append(cve)
        except Exception as e:
            print(f"Error loading {cve_file}: {e}")

print(f"Loaded {len(cves)} CVEs")

# Feature extraction
print("Extracting features...")
X = []
y = []

for cve in cves:
    features = []
    
    # Feature 1: CVE year (newer vulnerabilities more likely to be exploited)
    cve_year = 2020  # Default
    if "cve_id" in cve and re.match(r"CVE-\d+-\d+", cve["cve_id"]):
        try:
            cve_year = int(cve["cve_id"].split("-")[1])
        except:
            pass
    features.append(cve_year)
    
    # Feature 2: CVSS Base Score
    base_score = 0
    if "base_score" in cve:
        base_score = float(cve["base_score"])
    features.append(base_score)
    
    # Feature 3: Days since published
    days_since_published = 365  # Default to 1 year
    if "published" in cve and cve["published"]:
        try:
            published_date = datetime.fromisoformat(cve["published"].replace("Z", "+00:00"))
            days_since_published = (datetime.now() - published_date).days
        except:
            pass
    features.append(min(days_since_published, 1000))  # Cap at 1000 days
    
    # Target variable: is it exploited?
    is_exploited = 1 if cve.get("source") == "CISA KEV" or cve.get("exploited", False) else 0
    
    X.append(features)
    y.append(is_exploited)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Check if we have enough data
if len(X) < 10:
    print("Not enough data to train a model")
    exit(1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
print(f"Class distribution in training: {sum(y_train)} exploited, {len(y_train) - sum(y_train)} not exploited")

# Train Random Forest model
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Show feature importance
feature_names = ["CVE Year", "CVSS Base Score", "Days Since Published"]
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Ranking:")
for f in range(len(feature_names)):
    print(f"{f+1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")

# Save model and metadata
print("Saving model...")
with open("models/threat_model.pkl", "wb") as f:
    pickle.dump(model, f)

metadata = {
    "feature_names": feature_names,
    "training_date": datetime.now().isoformat(),
    "num_samples": len(X),
    "num_exploited": int(sum(y)),
    "accuracy": float(sum(y_pred == y_test) / len(y_test)),
    "feature_importance": {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}
}

with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Model training completed!")
'

echo "Model training completed at $(date)"
