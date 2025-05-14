#!/bin/bash
set -e

echo "🔧 Creating ultra-minimal Railway deployment fix..."

# 1. Create the simplest possible app.py that will pass healthcheck
cat > app.py << 'EOF'
from flask import Flask, jsonify

def create_app():
    app = Flask(__name__)
    
    @app.route('/', methods=['GET'])
    def index():
        return jsonify({'status': 'Echelon API is running'})
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({'status': 'healthy'})
    
    @app.route('/api/predictions', methods=['GET'])
    def get_predictions():
        return jsonify({'predictions': []})
    
    @app.route('/api/recent-predictions', methods=['GET'])
    def recent_predictions():
        return jsonify({'predictions': []})
    
    @app.route('/api/model-info', methods=['GET'])
    def model_info():
        return jsonify({
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'last_trained': 'never',
            'model_type': 'Placeholder'
        })
        
    return app

if __name__ == '__main__':
    app = create_app()
    from os import environ
    port = int(environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
EOF

# 2. Create minimal requirements.txt with only essential dependencies
cat > requirements.txt << 'EOF'
Flask==3.1.1
gunicorn==23.0.0
EOF

# 3. Create ultra-minimal railway.json
cat > railway.json << 'EOF'
{
  "build": {
    "builder": "nixpacks"
  },
  "deploy": {
    "startCommand": "gunicorn 'app:create_app()' --bind 0.0.0.0:$PORT",
    "healthcheckPath": "/api/health"
  }
}
EOF

# 4. Create simple railway_setup.sh
cat > railway_setup.sh << 'EOF'
#!/bin/bash
echo "Installing minimal dependencies..."
pip install -r requirements.txt
echo "Setup complete!"
EOF
chmod +x railway_setup.sh

echo "✅ Ultra-minimal deployment setup created!"
echo ""
echo "DEPLOYMENT INSTRUCTIONS:"
echo "------------------------"
echo "1. Install Railway CLI if not already installed:"
echo "   npm i -g @railway/cli"
echo ""
echo "2. Login to Railway:"
echo "   railway login"
echo ""
echo "3. Link to your Railway project (if not already linked):"
echo "   railway link"
echo ""
echo "4. Deploy the application:"
echo "   railway up"
echo ""
echo "IMPORTANT: After successful deployment, gradually replace this minimal version"
echo "with your actual functionality one component at a time until you find what's"
echo "causing the hanging issue."