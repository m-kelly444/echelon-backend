#!/bin/bash

# fix_ports.sh - Script to fix port configuration in Echelon deployment
echo "====================================================="
echo "ECHELON PORT CONFIGURATION FIX SCRIPT"
echo "====================================================="
echo "This script will update your code to use Railway's PORT environment variable"
echo "====================================================="

# Create backups of files
echo "Creating backups of files..."
cp api_server.py api_server.py.bak
echo "✓ Backup created: api_server.py.bak"

# Update api_server.py to use PORT environment variable
echo "Updating api_server.py to use environment variable PORT..."

# Check if os is already imported
if ! grep -q "^import os" api_server.py; then
    # Add os import if it doesn't exist
    sed -i '1,/^import/ s/^import/import os\nimport/' api_server.py
    echo "✓ Added import os to api_server.py"
fi

# Update PORT definition in api_server.py
if grep -q "^PORT =" api_server.py; then
    # Replace existing PORT definition
    sed -i 's/^PORT =.*/PORT = int(os.environ.get("PORT", 8000))/' api_server.py
    echo "✓ Updated PORT variable in api_server.py"
else
    # Add PORT definition after imports if it doesn't exist
    echo "⚠️ Could not find PORT variable definition. Please manually add this line after imports:"
    echo "PORT = int(os.environ.get(\"PORT\", 8000))"
fi

# Create wsgi.py
echo "Creating wsgi.py file..."
cat > wsgi.py << 'EOF'
#!/usr/bin/env python3
import os
from api_server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
EOF
echo "✓ Created wsgi.py"

# Create Procfile
echo "Creating Procfile..."
cat > Procfile << 'EOF'
web: python api_server.py
EOF
echo "✓ Created Procfile"

# Create railway.json
echo "Creating railway.json configuration..."
cat > railway.json << 'EOF'
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python api_server.py",
    "healthcheckPath": "/",
    "healthcheckTimeout": 300,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
EOF
echo "✓ Created railway.json"

# Git add, commit and push
echo "Adding and committing changes..."
git add api_server.py wsgi.py Procfile railway.json
git commit -m "Fix port configuration to use Railway PORT environment variable"
echo "✓ Changes committed"
echo ""
echo "Next steps:"
echo "1. Push changes to your repository: git push"
echo "2. Deploy to Railway: railway up"
echo "3. Get your deployment URL: railway status"
echo ""
echo "====================================================="
echo "PORT CONFIGURATION FIX COMPLETE"
echo "====================================================="