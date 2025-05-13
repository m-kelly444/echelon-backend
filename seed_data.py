import os
import json
import hashlib
import random
from datetime import datetime
from echelon.utils.logging import get_logger
from echelon.database import get_db_connection, init_db

logger = get_logger(__name__)

def generate_sample_threats(count=10):
    threats = []
    sources = ["SANS", "ThreatPost", "DarkReading", "ExploitDB"]
    attack_vectors = ["Phishing", "Ransomware", "Zero-day", "Supply Chain", "DDoS", "SQL Injection"]
    regions = ["North America", "Europe", "Asia", "Middle East"]
    sectors = ["Finance", "Healthcare", "Government", "Energy", "Technology"]
    
    for i in range(count):
        source = random.choice(sources)
        title = f"Sample threat {i+1} from {source}"
        description = f"This is a sample threat description for testing purposes. It simulates a {random.choice(attack_vectors)} attack targeting {random.choice(sectors)} in {random.choice(regions)}."
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
        
        threat['id'] = hashlib.sha256(f"{threat['title']}-{threat['source']}-{random.randint(1,1000)}".encode()).hexdigest()
        
        threats.append(threat)
    
    return threats

def store_sample_threats():
    threats = generate_sample_threats(20)
    stored_count = 0
    
    init_db()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        for threat in threats:
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

if __name__ == "__main__":
    logger.info("Seeding initial data...")
    count = store_sample_threats()
    logger.info(f"Added {count} sample threats to database")
