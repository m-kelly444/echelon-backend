#!/bin/bash
# Configuration variables for Echelon

# Data sources
CISA_KEV_URL="https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv"
MITRE_ENTERPRISE_URL="https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
NVD_API_URL="https://services.nvd.nist.gov/rest/json/cves/2.0?resultsPerPage=50"
RSS_FEEDS=(
  "https://www.cisa.gov/uscert/ncas/alerts.xml"
  "https://krebsonsecurity.com/feed/"
  "https://googleprojectzero.blogspot.com/feeds/posts/default"
)

# Database settings
DB_PATH="database/threats.db"

# API settings
API_PORT=8080
API_HOST="0.0.0.0"

# Model settings
MODEL_TYPE="random_forest"
FEATURE_IMPORTANCE_THRESHOLD=0.01
PREDICTION_THRESHOLD=0.7
MODEL_PATH="models/threat_model.pkl"
