{"type": "apt_group", "value": "APT41", "description": "Chinese state-sponsored group"},
        {"type": "apt_group", "value": "Lazarus Group", "description": "North Korean state-sponsored group"},
        {"type": "apt_group", "value": "Sandworm", "description": "Russian military unit targeting critical infrastructure"},
        {"type": "apt_group", "value": "Equation Group", "description": "Sophisticated group linked to NSA"},
        {"type": "apt_group", "value": "Darkhotel", "description": "Korean-speaking group targeting executives"},
        {"type": "apt_group", "value": "Carbanak", "description": "Financial crime group targeting banks"},
        
        {"type": "attack_type", "value": "Phishing", "description": "Social engineering attacks via email or messaging"},
        {"type": "attack_type", "value": "Ransomware", "description": "Malware that encrypts data and demands payment"},
        {"type": "attack_type", "value": "Zero-day", "description": "Exploits of previously unknown vulnerabilities"},
        {"type": "attack_type", "value": "Supply Chain", "description": "Attacks targeting the software supply chain"},
        {"type": "attack_type", "value": "DDoS", "description": "Distributed Denial of Service attacks"},
        {"type": "attack_type", "value": "Credential Theft", "description": "Stealing login credentials"},
        
        {"type": "threat_category", "value": "Social Engineering", "description": "Attacks exploiting human psychology"},
        {"type": "threat_category", "value": "Ransomware", "description": "Ransomware attacks"},
        {"type": "threat_category", "value": "Exploitation", "description": "Vulnerability exploitation"},
        {"type": "threat_category", "value": "Supply Chain", "description": "Supply chain compromises"},
        {"type": "threat_category", "value": "Data Exfiltration", "description": "Theft of sensitive data"},
        {"type": "threat_category", "value": "Insider Threat", "description": "Threats from within the organization"},
        
        {"type": "region", "value": "North America", "description": "North American region"},
        {"type": "region", "value": "Europe", "description": "European region"},
        {"type": "region", "value": "Asia", "description": "Asian region"},
        {"type": "region", "value": "Middle East", "description": "Middle Eastern region"},
        {"type": "region", "value": "South America", "description": "South American region"},
        {"type": "region", "value": "Africa", "description": "African region"},
        {"type": "region", "value": "Oceania", "description": "Oceania region"},
        
        {"type": "industry", "value": "Finance", "description": "Financial sector"},
        {"type": "industry", "value": "Healthcare", "description": "Healthcare sector"},
        {"type": "industry", "value": "Government", "description": "Government sector"},
        {"type": "industry", "value": "Technology", "description": "Technology sector"},
        {"type": "industry", "value": "Energy", "description": "Energy sector"},
        {"type": "industry", "value": "Retail", "description": "Retail sector"},
        {"type": "industry", "value": "Manufacturing", "description": "Manufacturing sector"},
        {"type": "industry", "value": "Defense", "description": "Defense sector"},
        
        {"type": "severity", "value": "High", "description": "High severity"},
        {"type": "severity", "value": "Medium", "description": "Medium severity"},
        {"type": "severity", "value": "Low", "description": "Low severity"}
    ]
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            for taxonomy in taxonomies:
                taxonomy_id = hashlib.sha256(f"{taxonomy['type']}-{taxonomy['value']}".encode()).hexdigest()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO taxonomy
                    (id, type, value, description, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        taxonomy_id,
                        taxonomy['type'],
                        taxonomy['value'],
                        taxonomy['description'],
                        now
                    )
                )
            conn.commit()
        
        logger.info(f"Initialized {len(taxonomies)} taxonomy values")
    except Exception as e:
        logger.error(f"Error initializing taxonomy: {str(e)}")

def collect_from_feeds(feeds):
    from echelon.data.manager import ThreatDataManager
    data_manager = ThreatDataManager()
    return data_manager.collect_data()

def collect_from_default_sources():
    default_sources = [
        "https://threatpost.com/feed/",
        "https://www.darkreading.com/rss.xml",
        "https://feeds.feedburner.com/TheHackersNews",
        "https://isc.sans.edu/rssfeed.xml",
        "https://www.welivesecurity.com/feed/",
        "https://securelist.com/feed/",
        "https://www.exploit-db.com/rss.xml"
    ]
    return collect_from_feeds(default_sources)

if __name__ == "__main__":
    seed_initial_data()
EOF

# Step 8: Create initialization script
cat > init_model.py << 'EOF'
import os
import json
import hashlib
from datetime import datetime
from echelon.utils.logging import get_logger
from echelon.database import init_db
from echelon.data.manager import ThreatDataManager
from echelon.ml.model import ThreatMLModel

logger = get_logger(__name__)

def main():
    logger.info("Initializing Echelon ML system...")
    
    init_db()
    logger.info("Database initialized")
    
    data_manager = ThreatDataManager()
    logger.info("Data manager initialized")
    
    ml_model = ThreatMLModel(data_manager)
    logger.info("ML model initialized")
    
    logger.info("Starting initial data collection...")
    data_manager.collect_data()
    
    logger.info("Starting initial model training...")
    training_result = ml_model.train()
    if training_result:
        logger.info("Initial model training completed successfully")
    else:
        logger.warning("Initial model training could not be completed")
    
    logger.info("Echelon ML system initialization complete")

if __name__ == "__main__":
    main()
EOF

# Step 9: Update app.py to use the ML implementation
cat > app.py << 'EOF'
from flask import Flask, jsonify, request
import os
import logging
import threading
import traceback

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import configurations
try:
    from config import SECRET_KEY, ALLOWED_ORIGINS
except Exception as e:
    logger.error(f"Error importing config: {str(e)}")
    SECRET_KEY = os.urandom(24).hex()
    ALLOWED_ORIGINS = ['*']

# Global component variables
data_manager = None
ml_model = None

def initialize_components():
    global data_manager, ml_model
    try:
        from echelon.database import init_db
        from echelon.data.manager import ThreatDataManager
        from echelon.ml.model import ThreatMLModel
        
        # Initialize database
        init_db()
        
        # Create components
        data_manager = ThreatDataManager()
        ml_model = ThreatMLModel(data_manager)
        
        logger.info("All components initialized successfully")
    except Exception as e:
        logger.error(f"Error in component initialization: {str(e)}")
        logger.error(traceback.format_exc())

def create_app(test_config=None):
    app = Flask(__name__)
    app.secret_key = SECRET_KEY
    
    # Start initialization in background
    threading.Thread(target=initialize_components, daemon=True).start()
    
    @app.after_request
    def after_request(response):
        origin = request.headers.get('Origin')
        if origin and (origin in ALLOWED_ORIGINS or '*' in ALLOWED_ORIGINS):
            response.headers.add('Access-Control-Allow-Origin', origin)
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        return response
    
    @app.route('/', methods=['GET'])
    def index():
        return jsonify({
            'status': 'Echelon Threat Intelligence API is running',
            'model_initialized': ml_model is not None,
            'data_manager_initialized': data_manager is not None
        })
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        # This endpoint must be FAST and not rely on any DB/ML initialization
        return jsonify({'status': 'healthy'})
    
    @app.route('/api/predictions', methods=['GET'])
    def get_predictions():
        try:
            # Ensure components are initialized
            global data_manager, ml_model
            if data_manager is None or ml_model is None:
                initialize_components()
                if data_manager is None or ml_model is None:
                    return jsonify({'error': 'System still initializing, please try again shortly'}), 503
            
            count = request.args.get('count', default=8, type=int)
            predictions = ml_model.predict_threats(num_predictions=count)
            
            # Store valid predictions
            stored_count = 0
            for prediction in predictions:
                if data_manager.store_prediction(prediction):
                    stored_count += 1
            
            return jsonify({
                'predictions': predictions,
                'count': len(predictions),
                'stored': stored_count
            })
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/recent-predictions', methods=['GET'])
    def recent_predictions():
        try:
            # Ensure data manager is initialized
            global data_manager
            if data_manager is None:
                initialize_components()
                if data_manager is None:
                    return jsonify({'error': 'System still initializing, please try again shortly'}), 503
            
            limit = request.args.get('limit', default=20, type=int)
            predictions = data_manager.get_recent_predictions(limit=limit)
            return jsonify({'predictions': predictions, 'count': len(predictions)})
        except Exception as e:
            logger.error(f"Error fetching recent predictions: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/model-info', methods=['GET'])
    def model_info():
        try:
            # Ensure ml model is initialized
            global ml_model
            if ml_model is None:
                initialize_components()
                if ml_model is None:
                    return jsonify({'error': 'System still initializing, please try again shortly'}), 503
            
            info = {
                'accuracy': getattr(ml_model, 'accuracy', 0),
                'precision': getattr(ml_model, 'precision', 0),
                'recall': getattr(ml_model, 'recall', 0),
                'f1': getattr(ml_model, 'f1', 0),
                'last_trained': getattr(ml_model, 'last_trained', 'never'),
                'model_type': 'Ensemble (BERT, RandomForest, GradientBoosting)'
            }
            return jsonify(info)
        except Exception as e:
            logger.error(f"Error fetching model info: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
            
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
EOF

# Step 10: Create a debug run script
cat > debug_run.sh << 'EOF'
#!/bin/bash
set -e

echo "🔬 Running Echelon ML in debug mode"

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export PYTHONPATH=$(pwd)
export LOG_LEVEL=DEBUG

# Create required directories
mkdir -p data models cache logs

# Initialize database with seed data if it doesn't exist
if [ ! -f "data/echelon.db" ]; then
    echo "🌱 Setting up database with initial data..."
    python seed_data.py
fi

# Run initial model training if needed
echo "🧠 Checking ML model initialization..."
python init_model.py

# Start the application
echo "🚀 Starting application..."
python -m gunicorn 'app:create_app()' --bind 0.0.0.0:5000 --workers 1 --timeout 300 --log-level debug
EOF
chmod +x debug_run.sh

# Make script executable
chmod +x fix_ml_implementation.sh

echo "✅ Echelon ML fix script created successfully!"
echo "Run ./fix_ml_implementation.sh to apply the changes to your project"
echo "Then run ./debug_run.sh to start the application with the new ML implementation"