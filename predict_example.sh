#!/bin/bash
# Example script to test the prediction API

echo "Testing Echelon prediction API..."

# Test CVE prediction
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"cve_id":"CVE-2023-1234","base_score":9.8}' \
  http://localhost:8080/predict

echo ""
echo "Done!"
