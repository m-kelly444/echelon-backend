import os
import json
import time
from datetime import datetime
import hashlib
import random

from echelon.utils.logging import get_logger
from echelon.database import get_db_connection
from echelon.data.manager import ThreatDataManager

logger = get_logger(__name__)

def generate_sample_threats(count=10):
    """Generate sample threat data for testing"""
    threats = []
    sources = ["SANS", "ThreatPost", "DarkReading", "ExploitDB"]
    attack_vectors = ["Phishing", "Ransomware", "Zero-day", "Supply Chain", "DDoS", "SQL Injection"]
    regions = ["North America", "Europe", "Asia", "Middle East"]
    sectors = ["Finance", "Healthcare", "Government", "Energy", "Technology"]
    
    for i in range(count):
        source = random.choice(sources)
        title = f"Sample threat {i+1} from {source}"
        description = f"This is a sample threat description for testing purposes. It simulates a threat from {source}."
        attack_vector = random.choice(attack_vectors)
        regions_affected = [random.choice(regions)]
        sectors_affected = [random.choice(sectors)]
        
        threat = {
            'source': source,
            'title': title,
            'description': description,
            'published_date': datetime.now().strftime("%Y-%m-%d"),
            'attack_vector': attack_vector,
            'regions_affected': regions_affected,
            'sectors_affected': sectors_affected,
            'severity': random.choice(["Low", "Medium", "High"]),
            'confidence': random.randint(60, 95),
            'tags': [],
            'indicators': {"ips": [], "urls": [], "cves": [], "hashes": []}
        }
        
        # Generate ID based on content
        threat['id'] = hashlib.sha256(f"{threat['title']}-{threat['source']}".encode()).hexdigest()
        
        threats.append(threat)
    
    return threats

def store_sample_threats():
    """Store sample threats in the database"""
    threats = generate_sample_threats(20)
    stored_count = 0
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        for threat in threats:
            # Check if threat already exists
            cursor.execute("SELECT id FROM threats WHERE id = ?", (threat['id'],))
            if not cursor.fetchone():
                cursor.execute(
                    """
                    INSERT INTO threats 
                    (id, source, title, description, published_date, tags, indicators, 
                     attack_vector, regions_affected, sectors_affected, 
                     severity, confidence, raw_data, processed, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        threat['id'],
                        threat['source'],
                        threat['title'],
                        threat['description'],
                        threat['published_date'],
                        json.dumps(threat.get('tags', [])),
                        json.dumps(threat.get('indicators', {})),
                        threat.get('attack_vector', ''),
                        json.dumps(threat.get('regions_affected', [])),
                        json.dumps(threat.get('sectors_affected', [])),
                        threat.get('severity', ''),
                        threat.get('confidence', 0),
                        '{}',
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                )
                stored_count += 1
                
                # Add to taxonomy as well
                if threat.get('attack_vector'):
                    attack_vector = threat['attack_vector']
                    attack_id = hashlib.sha256(f"attack-{attack_vector}".encode()).hexdigest()
                    
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO taxonomy 
                        (id, type, value, aliases, description, first_seen, last_updated, source, meta_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            attack_id,
                            'attack_type',
                            attack_vector,
                            json.dumps([]),
                            '',
                            datetime.now().strftime("%Y-%m-%d"),
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            threat.get('source', ''),
                            json.dumps({})
                        )
                    )
        
        conn.commit()
    
    return stored_count

def main():
    """Collect threat intelligence data"""
    logger.info("Starting manual data collection...")
    
    # For demonstration, we'll just add sample threats
    new_entries = store_sample_threats()
    
    logger.info(f"Data collection complete. Added {new_entries} new threat entries.")
    
    # Print summary of available data
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM threats")
        threat_count = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM taxonomy")
        taxonomy_count = cursor.fetchone()['count']
        
        logger.info(f"Database now contains {threat_count} threats and {taxonomy_count} taxonomy entries")

if __name__ == "__main__":
    main()
