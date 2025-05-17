#!/bin/bash

# echelon_railway_deploy.sh - Deploy Echelon to Railway

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}ECHELON RAILWAY DEPLOYMENT SCRIPT${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo -e "${YELLOW}This script will prepare and deploy Echelon to Railway${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Check if necessary tools are installed
echo -e "\n${YELLOW}Checking required tools...${NC}"

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}Git is not installed. Please install Git before proceeding.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Git is installed${NC}"

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo -e "${YELLOW}Railway CLI not found. Installing Railway CLI...${NC}"
    npm i -g @railway/cli
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install Railway CLI. Please install it manually.${NC}"
        echo -e "${YELLOW}You can install it with: npm i -g @railway/cli${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Railway CLI installed${NC}"
else
    echo -e "${GREEN}✓ Railway CLI is installed${NC}"
fi

# Login to Railway
echo -e "\n${YELLOW}Logging in to Railway (you'll need to authenticate in browser)...${NC}"
railway login
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to login to Railway. Please try again.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Successfully logged in to Railway${NC}"

# Create runtime.txt to specify Python version
echo -e "\n${YELLOW}Creating runtime.txt to specify Python 3.9...${NC}"
echo "python-3.9.18" > runtime.txt
echo -e "${GREEN}✓ Created runtime.txt with Python 3.9${NC}"

# Create requirements.txt with compatible versions
echo -e "\n${YELLOW}Creating requirements.txt with compatible versions...${NC}"
cat > requirements.txt << 'EOF'
numpy==1.23.5
scipy==1.9.3
scikit-learn==1.0.2
requests==2.28.2
tenacity==8.1.0
pycountry==22.3.5
flask==2.2.3
flask_cors==3.0.10
gunicorn==20.1.0
EOF
echo -e "${GREEN}✓ Created requirements.txt with compatible versions${NC}"

# Create Procfile for Railway
echo -e "\n${YELLOW}Creating Procfile for Railway...${NC}"
cat > Procfile << 'EOF'
web: gunicorn --preload --workers=2 "api_server:create_app()"
EOF
echo -e "${GREEN}✓ Created Procfile${NC}"

# Modify api_server.py to work with Railway and Gunicorn
echo -e "\n${YELLOW}Modifying api_server.py for Railway and Gunicorn compatibility...${NC}"
# Create a backup first
cp api_server.py api_server.py.bak.$(date +%Y%m%d%H%M%S)

# Create a wrapper function at the end of the file
cat >> api_server.py << 'EOF'

# Function to create app for Gunicorn
def create_app():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    
    # Use Railway's port or default to 8080
    port = int(os.environ.get('PORT', 8080))
    
    # Create and return the server
    return HTTPServer((HOST, port), PredictionHandler)

# Modify main block to use the port from environment
if __name__ == "__main__":
    # Use Railway's port
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting ML attack forecasting server on {HOST}:{port}")
    
    server = HTTPServer((HOST, port), PredictionHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped by user")
        refresh_handler.stop_monitoring()
        server.server_close()
EOF
echo -e "${GREEN}✓ Modified api_server.py for Railway compatibility${NC}"

# Create .gitignore if it doesn't exist
echo -e "\n${YELLOW}Creating .gitignore file...${NC}"
if [ ! -f ".gitignore" ]; then
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Railway specific
.railway/

# Logs
logs/
*.log

# Backup files
*.bak
*.bak.*

# Data files that are too large or sensitive
data/raw/*/
!data/raw/.gitkeep
data/processed/*/
!data/processed/.gitkeep

# Allow some necessary data directories
!data/raw/cisa/.gitkeep
!data/raw/mitre/.gitkeep
!data/raw/nvd/.gitkeep
!data/raw/otx/.gitkeep
!data/raw/abuseipdb/.gitkeep
!data/processed/cves/.gitkeep
!data/processed/apt/.gitkeep
!data/processed/geo/.gitkeep

# Models
models/*.pkl
EOF
    echo -e "${GREEN}✓ Created .gitignore${NC}"
else
    echo -e "${GREEN}✓ .gitignore already exists${NC}"
fi

# Create data directories if they don't exist
echo -e "\n${YELLOW}Setting up data directories...${NC}"
mkdir -p data/raw/{cisa,mitre,nvd,otx,abuseipdb,rss}
mkdir -p data/processed/{cves,apt,geo,techniques,alerts}
mkdir -p models
mkdir -p logs

# Create .gitkeep files to track empty directories
for dir in data/raw/{cisa,mitre,nvd,otx,abuseipdb,rss} data/processed/{cves,apt,geo,techniques,alerts} models logs; do
    touch "$dir/.gitkeep"
done

echo -e "${GREEN}✓ Data directories set up${NC}"

# Ensure config directory and API keys file exist
echo -e "\n${YELLOW}Checking API configuration...${NC}"
mkdir -p config
if [ ! -f "config/api_keys.json" ]; then
    echo -e "${YELLOW}Creating template API keys file...${NC}"
    cat > config/api_keys.json << 'EOF'
{
  "alienvault_otx": {
    "api_key": "YOUR_ALIENVAULT_OTX_KEY"
  },
  "abuseipdb": {
    "api_key": "YOUR_ABUSEIPDB_KEY"
  },
  "mitre_attack": {
    "url": "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
  },
  "cisa_kev": {
    "url": "https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv"
  }
}
EOF
    echo -e "${YELLOW}Created template API keys file. You'll need to set these as environment variables in Railway.${NC}"
else
    echo -e "${GREEN}✓ API configuration file already exists${NC}"
fi

# Create railway.json configuration
echo -e "\n${YELLOW}Creating Railway configuration file...${NC}"
cat > railway.json << 'EOF'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS",
    "buildCommand": ""
  },
  "deploy": {
    "startCommand": "gunicorn --preload --workers=2 \"api_server:create_app()\"",
    "healthcheckPath": "/",
    "healthcheckTimeout": 300,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
EOF
echo -e "${GREEN}✓ Created Railway configuration${NC}"

# Create a basic wsgi.py file for Gunicorn
echo -e "\n${YELLOW}Creating wsgi.py file for Gunicorn...${NC}"
cat > wsgi.py << 'EOF'
from api_server import create_app

# This is used by Gunicorn
app = create_app()
EOF
echo -e "${GREEN}✓ Created wsgi.py${NC}"

# Initialize Git if needed
echo -e "\n${YELLOW}Setting up Git repository...${NC}"
if [ ! -d ".git" ]; then
    git init
    echo -e "${GREEN}✓ Initialized Git repository${NC}"
else
    echo -e "${GREEN}✓ Git repository already initialized${NC}"
fi

# Create a .reload_data file for the data refresh mechanism
echo -e "\n${YELLOW}Creating .reload_data file for data refresh mechanism...${NC}"
touch .reload_data
echo -e "${GREEN}✓ Created .reload_data file${NC}"

# Add and commit changes
echo -e "\n${YELLOW}Adding and committing changes...${NC}"
git add .
git commit -m "Railway deployment setup with Python 3.9"
echo -e "${GREEN}✓ Changes committed${NC}"

# Initialize Railway project
echo -e "\n${YELLOW}Initializing Railway project...${NC}"
railway init
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to initialize Railway project. Please try again.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Railway project initialized${NC}"

# Prompt for API keys to set as environment variables
echo -e "\n${YELLOW}Setting up environment variables for Railway...${NC}"
echo -e "${BLUE}Do you want to set API keys as environment variables now? (y/n)${NC}"
read set_keys

if [ "$set_keys" = "y" ] || [ "$set_keys" = "Y" ]; then
    echo -e "${YELLOW}Enter your AlienVault OTX API key (leave blank to skip):${NC}"
    read otx_key
    if [ ! -z "$otx_key" ]; then
        railway variables set ALIENVAULT_OTX_API_KEY="$otx_key"
        echo -e "${GREEN}✓ Set ALIENVAULT_OTX_API_KEY${NC}"
    fi
    
    echo -e "${YELLOW}Enter your AbuseIPDB API key (leave blank to skip):${NC}"
    read abuse_key
    if [ ! -z "$abuse_key" ]; then
        railway variables set ABUSEIPDB_API_KEY="$abuse_key"
        echo -e "${GREEN}✓ Set ABUSEIPDB_API_KEY${NC}"
    fi
    
    # Add environment variables to access API keys
    echo -e "${YELLOW}Setting environment variables to access API keys...${NC}"
    railway variables set PYTHON_VERSION="3.9.18"
    echo -e "${GREEN}✓ Set PYTHON_VERSION to 3.9.18${NC}"
else
    echo -e "${YELLOW}Skipping environment variables setup.${NC}"
    echo -e "${YELLOW}Remember to set these manually in the Railway dashboard:${NC}"
    echo -e "${BLUE}  - ALIENVAULT_OTX_API_KEY${NC}"
    echo -e "${BLUE}  - ABUSEIPDB_API_KEY${NC}"
    echo -e "${BLUE}  - PYTHON_VERSION=3.9.18${NC}"
fi

# Deploy to Railway
echo -e "\n${YELLOW}Deploying to Railway...${NC}"
railway up
if [ $? -ne 0 ]; then
    echo -e "${RED}Deployment to Railway failed. Please check the error message above.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Successfully deployed to Railway!${NC}"

# Get the deployed URL
echo -e "\n${YELLOW}Getting your Railway deployment URL...${NC}"
railway status

# Display instructions for Vercel integration
echo -e "\n${GREEN}=======================================================${NC}"
echo -e "${GREEN}DEPLOYMENT SUCCESSFUL!${NC}"
echo -e "${GREEN}=======================================================${NC}"
echo -e "${YELLOW}To integrate with your Vercel frontend:${NC}"
echo -e ""
echo -e "${BLUE}1. Add your Railway URL as an environment variable in Vercel:${NC}"
echo -e "   NEXT_PUBLIC_API_URL=https://your-railway-app-url"
echo -e ""
echo -e "${BLUE}2. In your Vercel frontend, use this variable for API calls:${NC}"
echo -e "   const response = await fetch(\`\${process.env.NEXT_PUBLIC_API_URL}/dashboard/overview\`);"
echo -e ""
echo -e "${BLUE}3. To view available endpoints, visit:${NC}"
echo -e "   https://your-railway-app-url/"
echo -e ""
echo -e "${YELLOW}IMPORTANT: If you didn't set API keys as environment variables,${NC}"
echo -e "${YELLOW}add them in the Railway dashboard before using the API.${NC}"
echo -e "${GREEN}=======================================================${NC}"

# Finish
echo -e "\n${GREEN}Deployment script complete!${NC}"
exit