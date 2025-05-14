import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv('FLASK_SECRET_KEY', os.urandom(24).hex())
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'https://final-omega-drab.vercel.app,http://localhost:3000').split(',')

DB_PATH = os.getenv('DB_PATH', 'data/echelon.db')
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
CACHE_DIR = os.getenv('CACHE_DIR', 'cache')
DATA_DIR = os.getenv('DATA_DIR', 'data')

for directory in [MODEL_DIR, CACHE_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'logs/echelon.log')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

CONFIDENCE_THRESHOLD = int(os.getenv('CONFIDENCE_THRESHOLD', 70))
PREDICTION_HORIZON_DAYS = int(os.getenv('PREDICTION_HORIZON_DAYS', 7))
