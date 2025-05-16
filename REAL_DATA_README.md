# Echelon - Real Data Only Mode

This installation of Echelon has been configured to use only real threat intelligence data, with all fallbacks, synthetic data, and mocks removed.

## Requirements

To run Echelon in real-data-only mode, you need:

1. **API Keys**: Valid API keys for:
   - AlienVault OTX
   - AbuseIPDB

2. **Real Data Sources**: The system must have already collected data from:
   - CISA KEV
   - MITRE ATT&CK
   - NVD
   - AlienVault OTX
   - AbuseIPDB

## Setup Instructions

1. Configure your API keys in `config/api_keys.json`

2. Run the data collection script:
./enhance_backend.sh

3. Validate that all real data is available:
python3 validate_real_data.py

4. Start the system in real-data-only mode:
./run_real_data_only.sh

## Troubleshooting

If the validation fails, ensure:

1. Your API keys are correctly configured
2. The data collection process has been run successfully
3. All required directories exist and have proper permissions
4. Internet connectivity is available for data collection

## Data Flow

In real-data-only mode, Echelon:

1. Uses only verified threat intelligence from authoritative sources
2. Makes predictions only when real APT mappings exist
3. Shows "Data Unavailable" rather than using synthetic or mock data
4. Never falls back to heuristic-based matching or guessing

## API Endpoints

- `/` - API information
- `/predict` - Make a prediction (POST, requires real data)
- `/status` - Check system status
- `/cves` - List real CVEs
- `/geo` - Get geographic threat data from real sources
