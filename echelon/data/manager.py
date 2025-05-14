import json
import hashlib
from datetime import datetime
from echelon.database import get_db_connection

class ThreatDataManager:
    def __init__(self):
        pass
    
    def get_taxonomy_values(self, type_name):
        return []
    
    def store_prediction(self, prediction):
        try:
            if 'id' not in prediction:
                prediction['id'] = hashlib.sha256(f"{prediction.get('apt_group', 'unknown')}-{prediction.get('timestamp', datetime.now().isoformat())}".encode()).hexdigest()
            
            if 'created_at' not in prediction:
                prediction['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO predictions 
                    (id, apt_group, attack_type, threat_category, region, industry, severity, likelihood, 
                    confidence, timestamp, description, indicators, affecting, evidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        prediction['id'],
                        prediction.get('apt_group', ''),
                        prediction.get('attack_type', ''),
                        prediction.get('threat_category', ''),
                        prediction.get('region', ''),
                        prediction.get('industry', ''),
                        prediction.get('severity', ''),
                        prediction.get('likelihood', ''),
                        prediction.get('confidence', 0),
                        prediction.get('timestamp', ''),
                        prediction.get('description', ''),
                        json.dumps(prediction.get('indicators', {})),
                        prediction.get('affecting', ''),
                        json.dumps(prediction.get('evidence', [])),
                        prediction['created_at']
                    )
                )
                conn.commit()
            return True
        except Exception:
            return False
    
    def get_recent_predictions(self, limit=20):
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM predictions 
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """, 
                    (limit,)
                )
                predictions = cursor.fetchall()
            
            for prediction in predictions:
                prediction['indicators'] = json.loads(prediction['indicators'])
                prediction['evidence'] = json.loads(prediction['evidence'])
            
            return predictions
        except Exception:
            return []
