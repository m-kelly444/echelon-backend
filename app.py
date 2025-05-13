from flask import Flask, jsonify
import threading
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
        header = response.headers
        origin = request.headers.get('Origin')
        if origin in ALLOWED_ORIGINS:
            header['Access-Control-Allow-Origin'] = origin
        header['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
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
    
    return app

if __name__ == '__main__':
    app = create_app()
    from flask import request
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
