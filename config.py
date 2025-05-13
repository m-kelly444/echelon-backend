import os
import dotenv
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "Echelon"
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
TESTING = os.getenv('TESTING', 'False').lower() == 'true'
SECRET_KEY = os.getenv('FLASK_SECRET_KEY', os.urandom(24).hex())
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'https://final-omega-drab.vercel.app,http://localhost:3000').split(',')

DB_PATH = os.getenv('DB_PATH', 'data/echelon.db')
SQLITE_URI = f"sqlite:///{DB_PATH}"

MODEL_DIR = os.getenv('MODEL_DIR', 'models')
CACHE_DIR = os.getenv('CACHE_DIR', 'cache')
DATA_DIR = os.getenv('DATA_DIR', 'data')

for directory in [MODEL_DIR, CACHE_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, "threat_model.pt")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
ENCODERS_FILE = os.path.join(MODEL_DIR, "encoders.joblib")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.joblib")

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'logs/echelon.log')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

DATA_COLLECTION_INTERVAL_HOURS = int(os.getenv('DATA_COLLECTION_INTERVAL_HOURS', 6))
PREDICTION_HORIZON_DAYS = int(os.getenv('PREDICTION_HORIZON_DAYS', 7))
TRAINING_INTERVAL_HOURS = int(os.getenv('TRAINING_INTERVAL_HOURS', 12))

MAX_FEATURES = int(os.getenv('MAX_FEATURES', 1000))
HIDDEN_DIM = int(os.getenv('HIDDEN_DIM', 256))
CONFIDENCE_THRESHOLD = int(os.getenv('CONFIDENCE_THRESHOLD', 70))

SECURITY_FEEDS = os.getenv('SECURITY_FEEDS', 'https://threatpost.com/feed/,https://www.darkreading.com/rss.xml,https://feeds.feedburner.com/TheHackersNews').split(',')
THREAT_FEEDS = os.getenv('THREAT_FEEDS', 'https://isc.sans.edu/rssfeed.xml,https://www.welivesecurity.com/feed/,https://securelist.com/feed/').split(',')
EXPLOITDB_FEEDS = os.getenv('EXPLOITDB_FEEDS', 'https://www.exploit-db.com/rss.xml').split(',')

MITRE_ATTACK_URL = os.getenv('MITRE_ATTACK_URL', 'https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json')
CISA_KEV_URL = os.getenv('CISA_KEV_URL', 'https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json')
