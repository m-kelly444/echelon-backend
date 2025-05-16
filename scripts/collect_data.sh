#!/bin/bash
# Script to collect threat intelligence data from various sources

source config.sh

# Create data directories if they don't exist
mkdir -p data/raw/{cisa,mitre,nvd,rss}

echo "Starting data collection at $(date)"

# Download CISA KEV data
echo "Downloading CISA KEV data..."
curl -s -L "$CISA_KEV_URL" -o data/raw/cisa/kev.csv
if [ $? -eq 0 ] && [ -s "data/raw/cisa/kev.csv" ]; then
  echo "  ✓ CISA KEV data downloaded successfully"
else
  echo "  ✗ Failed to download CISA KEV data"
fi

# Download MITRE ATT&CK data
echo "Downloading MITRE ATT&CK data..."
curl -s -L "$MITRE_ENTERPRISE_URL" -o data/raw/mitre/enterprise.json
if [ $? -eq 0 ] && [ -s "data/raw/mitre/enterprise.json" ]; then
  echo "  ✓ MITRE ATT&CK data downloaded successfully"
else
  echo "  ✗ Failed to download MITRE ATT&CK data"
fi

# Download NVD data
echo "Downloading NVD data..."
curl -s -L "$NVD_API_URL" -o data/raw/nvd/recent.json
if [ $? -eq 0 ] && [ -s "data/raw/nvd/recent.json" ]; then
  echo "  ✓ NVD data downloaded successfully"
else
  echo "  ✗ Failed to download NVD data"
fi

# Download RSS feeds
echo "Downloading RSS feeds..."
for feed_url in "${RSS_FEEDS[@]}"; do
  feed_name=$(echo "$feed_url" | md5sum | cut -d' ' -f1)
  echo "  Downloading feed: $feed_url"
  curl -s -L "$feed_url" -o "data/raw/rss/$feed_name.xml"
  if [ $? -eq 0 ] && [ -s "data/raw/rss/$feed_name.xml" ]; then
    echo "    ✓ Feed downloaded successfully"
  else
    echo "    ✗ Failed to download feed"
  fi
done

echo "Data collection completed at $(date)"
