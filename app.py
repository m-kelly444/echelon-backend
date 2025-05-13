from flask import Flask, jsonify, request
import threading
import os
from echelon.utils.logging import get_logger
from echelon.database import init_db
from echelon.data.manager import ThreatDataManager
from echelon.ml.model import ThreatMLModel
from config import SECRET_KEY, ALLOWED_ORIGINS
import json

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
