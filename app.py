from flask import Flask, jsonify, request
import os
import logging
import threading

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import configurations but catch any errors
try:
    from config import SECRET_KEY, ALLOWED_ORIGINS
except Exception as e:
    logger.error(f"Error importing config: {str(e)}")
    SECRET_KEY = os.urandom(24).hex()
    ALLOWED_ORIGINS = ['*']

# Initialize empty database structures
def lazy_init_db():
    try:
        from echelon.database import init_db
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")

# Create thread for lazy loading components
def lazy_load_components():
    try:
        # Perform heavy initializations here
        from echelon.database import init_db
        from echelon.data.manager import ThreatDataManager
        from echelon.ml.model import ThreatMLModel
        
        # Initialize database
        init_db()
        
        # Create basic components
        data_manager = ThreatDataManager()
        ml_model = ThreatMLModel(data_manager)
        
        logger.info("All components loaded successfully")
    except Exception as e:
        logger.error(f"Error in lazy loading: {str(e)}")

# Global component variables
data_manager = None
ml_model = None

def create_app(test_config=None):
    app = Flask(__name__)
    app.secret_key = SECRET_KEY
    
    # Start lazy loading in background
    threading.Thread(target=lazy_load_components, daemon=True).start()
    
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
        return jsonify({'status': 'Echelon Threat Intelligence API is running'})
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        # This endpoint must be FAST and not rely on any DB/ML initialization
        return jsonify({'status': 'healthy'})
    
    @app.route('/api/predictions', methods=['GET'])
    def get_predictions():
        try:
            # Lazy initialize components if needed
            global data_manager, ml_model
            if data_manager is None or ml_model is None:
                from echelon.data.manager import ThreatDataManager
                from echelon.ml.model import ThreatMLModel
                data_manager = ThreatDataManager()
                ml_model = ThreatMLModel(data_manager)
            
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
            # Lazy initialize data manager if needed
            global data_manager
            if data_manager is None:
                from echelon.data.manager import ThreatDataManager
                data_manager = ThreatDataManager()
            
            limit = request.args.get('limit', default=20, type=int)
            predictions = data_manager.get_recent_predictions(limit=limit)
            return jsonify({'predictions': predictions})
        except Exception as e:
            logger.error(f"Error fetching recent predictions: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/model-info', methods=['GET'])
    def model_info():
        try:
            # Lazy initialize model if needed
            global ml_model
            if ml_model is None:
                from echelon.data.manager import ThreatDataManager
                from echelon.ml.model import ThreatMLModel
                data_manager = ThreatDataManager()
                ml_model = ThreatMLModel(data_manager)
            
            info = {
                'accuracy': getattr(ml_model, 'accuracy', 0),
                'precision': getattr(ml_model, 'precision', 0),
                'recall': getattr(ml_model, 'recall', 0),
                'f1': getattr(ml_model, 'f1', 0),
                'last_trained': getattr(ml_model, 'last_trained', 'never'),
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
