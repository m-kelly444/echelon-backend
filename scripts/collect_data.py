import os
import json
import certifi
import feedparser
import requests
from datetime import datetime
from echelon.utils.logging import get_logger
from echelon.database import get_db_connection
from echelon.data.manager import ThreatDataManager

os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

logger = get_logger(__name__)

def main():
    logger.info("Starting manual data collection...")
    
    data_manager = ThreatDataManager()
    
    new_entries = data_manager.collect_data()
    
    logger.info(f"Data collection complete. Added {new_entries} new threat entries.")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM threats")
        threat_count = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM taxonomy")
        taxonomy_count = cursor.fetchone()['count']
        
        logger.info(f"Database now contains {threat_count} threats and {taxonomy_count} taxonomy entries")

if __name__ == "__main__":
    main()
