#!/bin/bash
set -e

# Fix requirements
cat > requirements.txt << 'END'
Flask==3.1.1
gunicorn==23.0.0
python-dotenv==1.1.0
numpy>=1.20.0
pandas>=1.3.0
requests>=2.25.0
scikit-learn>=1.0.0
joblib>=1.0.0
END

# Fix setup.py
cat > setup.py << 'END'
from setuptools import setup, find_packages

setup(
    name="echelon",
    version="0.1",
    packages=find_packages()
)
END

# Fix railway.json
cat > railway.json << 'END'
{
  "build": {
    "builder": "nixpacks"
  },
  "deploy": {
    "startCommand": "gunicorn 'app:create_app()' --bind 0.0.0.0:$PORT --workers 1 --timeout 120",
    "healthcheckPath": "/api/health",
    "healthcheckTimeout": 300
  }
}
END

# Make sure Procfile is correct
cat > Procfile << 'END'
web: gunicorn 'app:create_app()' --bind 0.0.0.0:$PORT --workers 1 --timeout 120
END

# Remove package.json to avoid confusion
rm -f package.json

# Confirm changes
echo "Files updated for deployment"
