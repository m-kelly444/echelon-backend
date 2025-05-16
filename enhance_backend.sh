#!/bin/bash
# Main script to enhance Echelon backend with real API data

echo "========================================="
echo "ECHELON: ENHANCING BACKEND WITH REAL DATA"
echo "========================================="

# Check Python dependencies
echo "Checking Python dependencies..."
REQUIRED_PACKAGES=("requests" "numpy" "pycountry" "scikit-learn")
MISSING_PACKAGES=()

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    python3 -c "import $pkg" 2>/dev/null
    if [ $? -ne 0 ]; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "Installing missing Python packages: ${MISSING_PACKAGES[*]}"
    pip3 install ${MISSING_PACKAGES[*]}
fi

# Create needed directories
mkdir -p config
mkdir -p models/apt_attribution
mkdir -p data/processed/geo
mkdir -p data/processed/apt
mkdir -p data/raw/otx
mkdir -p data/raw/abuseipdb

# Collect real data
echo "Collecting real threat intelligence data..."
python3 scripts/advanced_data_collector.py

# Apply patches to API server
echo "Enhancing API server with new capabilities..."
python3 patches/enhance_api_server.py

# Create example startup script
cat > run_enhanced_api.sh << 'EOT'
#!/bin/bash
# Run enhanced Echelon API server
echo "Starting enhanced Echelon API server..."
python3 api_server.py
EOT
chmod +x run_enhanced_api.sh

echo "========================================="
echo "Backend enhancement complete!"
echo "========================================="
echo "The following enhancements have been made:"
echo " - Added real API data integration with AlienVault OTX and AbuseIPDB"
echo " - Added APT group attribution based on real threat intelligence"
echo " - Added attack type prediction"
echo " - Added geographic data processing"
echo " - Enhanced API server to support all new features"
echo ""
echo "To run the enhanced API server:"
echo "  ./run_enhanced_api.sh"
echo ""
echo "To manually update threat data:"
echo "  python3 scripts/advanced_data_collector.py"
