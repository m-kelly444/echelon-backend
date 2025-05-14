#!/bin/bash

# echelon_railway_fix.sh - Configure Echelon backend for Railway deployment

echo "🚀 Echelon Railway Backend Setup"
echo "-------------------------------"

# Create required directories
echo "📁 Creating directory structure..."
mkdir -p models data cache logs
touch logs/echelon.log

# Update environment variables for Railway
echo "🔧 Setting up Railway environment configuration..."
cat > .env.railway << 'END_ENV'
# Flask settings
FLASK_SECRET_KEY=railway_secret_key_for_echelon_app_$(openssl rand -hex 8)
DEBUG=false
TESTING=false

# CORS settings
ALLOWED_ORIGINS=https://final-omega-drab.vercel.app,http://localhost:3000

# Database settings
DB_PATH=data/echelon.db

# Directory settings
MODEL_DIR=models
CACHE_DIR=cache
DATA_DIR=data

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/echelon.log

# Model settings - will be read from config.py
END_ENV

# Create Railway configuration files
echo "🚆 Creating Railway deployment configuration..."
cat > railway.json << 'END_RAILWAY_JSON'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS",
    "buildCommand": "chmod +x railway_setup.sh && ./railway_setup.sh"
  },
  "deploy": {
    "startCommand": "gunicorn 'app:create_app()' --bind 0.0.0.0:$PORT --workers 1 --timeout 120",
    "healthcheckPath": "/api/health",
    "healthcheckTimeout": 300
  }
}
END_RAILWAY_JSON

# Create Railway setup script
echo "🔧 Creating Railway setup script..."
cat > railway_setup.sh << 'END_SETUP'
#!/bin/bash
set -e

echo "🔨 Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "🌱 Setting up initial database..."
python seed_data.py

echo "✅ Build completed successfully!"
END_SETUP

chmod +x railway_setup.sh

# Fix app.py for proper CORS handling
echo "🌐 Updating CORS settings in app.py..."
cat > app.py << 'END_APP'
from flask import Flask, jsonify, request
import os
from echelon.utils.logging import get_logger
from echelon.database import init_db
from echelon.data.manager import ThreatDataManager
from echelon.ml.model import ThreatMLModel
from config import SECRET_KEY, ALLOWED_ORIGINS

logger = get_logger(__name__)

def create_app(test_config=None):
    app = Flask(__name__)
    app.secret_key = SECRET_KEY
    
    @app.after_request
    def after_request(response):
        origin = request.headers.get('Origin')
        if origin and (origin in ALLOWED_ORIGINS or '*' in ALLOWED_ORIGINS):
            response.headers.add('Access-Control-Allow-Origin', origin)
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response
    
    init_db()
    data_manager = ThreatDataManager()
    ml_model = ThreatMLModel(data_manager)
    
    @app.route('/', methods=['GET'])
    def index():
        return jsonify({'status': 'Echelon Threat Intelligence API is running'})
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'healthy'})
    
    @app.route('/api/predictions', methods=['GET'])
    def get_predictions():
        try:
            count = request.args.get('count', default=8, type=int)
            predictions = ml_model.predict_threats(num_predictions=count)
            for prediction in predictions:
                data_manager.store_prediction(prediction)
            return jsonify({'predictions': predictions})
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/recent-predictions', methods=['GET'])
    def recent_predictions():
        try:
            limit = request.args.get('limit', default=20, type=int)
            predictions = data_manager.get_recent_predictions(limit=limit)
            return jsonify({'predictions': predictions})
        except Exception as e:
            logger.error(f"Error fetching recent predictions: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/model-info', methods=['GET'])
    def model_info():
        try:
            info = {
                'accuracy': ml_model.accuracy,
                'precision': ml_model.precision,
                'recall': ml_model.recall,
                'f1': ml_model.f1,
                'last_trained': ml_model.last_trained,
                'model_type': 'Neural Network'
            }
            return jsonify(info)
        except Exception as e:
            logger.error(f"Error fetching model info: {str(e)}")
            return jsonify({'error': str(e)}), 500
            
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
END_APP

# Create a comprehensive debug script
echo "🐞 Creating debug script..."
cat > debug_run.sh << 'END_DEBUG'
#!/bin/bash
set -e

# Explicitly set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export PYTHONPATH=$(pwd)
export LOG_LEVEL=DEBUG

# Create required directories
mkdir -p data models cache logs

# Initialize database with seed data if it doesn't exist
if [ ! -f "data/echelon.db" ]; then
    echo "🌱 Setting up database with sample data..."
    python seed_data.py
fi

# Start the application with verbose output
echo "🚀 Starting application..."
python -m gunicorn 'app:create_app()' --bind 0.0.0.0:5000 --workers 1 --timeout 120 --log-level debug
END_DEBUG
chmod +x debug_run.sh

# Create Procfile for deployment
echo "🚀 Creating Procfile for deployment..."
cat > Procfile << 'END_PROCFILE'
web: gunicorn 'app:create_app()' --bind 0.0.0.0:$PORT --workers 1 --timeout 120
END_PROCFILE

# Update requirements.txt to ensure essential packages
echo "📦 Updating requirements.txt with essential packages..."
cat >> requirements.txt << 'END_REQUIREMENTS'
# Additional packages if not already in requirements.txt
Flask==3.1.1
gunicorn==23.0.0
python-dotenv==1.1.0
numpy>=1.20.0
pandas>=1.3.0
torch>=2.0.0
requests>=2.25.0
scikit-learn>=1.0.0
joblib>=1.0.0
END_REQUIREMENTS

echo "✅ Echelon Railway setup complete! Your backend is now ready for deployment."
