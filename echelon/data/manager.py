import json
import hashlib
import threading
import schedule
import time
import re
import certifi
import os
import feedparser
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from echelon.utils.logging import get_logger
from echelon.database import get_db_connection, execute_query
from config import (
    SECURITY_FEEDS, THREAT_FEEDS, EXPLOITDB_FEEDS, DATA_COLLECTION_INTERVAL_HOURS,
    CISA_KEV_URL, MITRE_ATTACK_URL
)

logger = get_logger(__name__)

os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

PATTERN_IP = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
PATTERN_URL = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+')
PATTERN_CVE = re.compile(r'CVE-\d{4}-\d{4,7}')
PATTERN_HASH = re.compile(r'\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b')

class ThreatDataManager:
    def __init__(self):
        self.sources = self._initialize_sources()
        self.db_lock = threading.Lock()
        self.setup_scheduled_collection()
    
    def _initialize_sources(self):
        sources = []
        
        if SECURITY_FEEDS:
            sources.append(("SecurityNews", SECURITY_FEEDS))
        
        if THREAT_FEEDS:
            sources.append(("ThreatResearch", THREAT_FEEDS))
        
        if EXPLOITDB_FEEDS:
            sources.append(("ExploitDB", EXPLOITDB_FEEDS))
        
        return sources
    
    def collect_data(self):
        logger.info("Starting data collection process")
        total_new_entries = 0
        all_data = []
        
        with ThreadPoolExecutor(max_workers=len(self.sources)) as executor:
            futures = {executor.submit(self._fetch_feed_data, source_name, urls): source_name 
                      for source_name, urls in self.sources}
            
            for future in futures:
                try:
                    data = future.result()
                    all_data.extend(data)
                except Exception as e:
                    source_name = futures[future]
                    logger.error(f"Error collecting data from {source_name}: {str(e)}")
        
        logger.info(f"Collected {len(all_data)} total entries from all sources")
        
        with self.db_lock:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                for item in all_data:
                    cursor.execute("SELECT id FROM threats WHERE id = ?", (item['id'],))
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
                                item['id'],
                                item['source'],
                                item['title'],
                                item['description'],
                                item['published_date'],
                                json.dumps(item.get('tags', [])),
                                json.dumps(item.get('indicators', {})),
                                item.get('attack_vector', ''),
                                json.dumps(item.get('regions_affected', [])),
                                json.dumps(item.get('sectors_affected', [])),
                                item.get('severity', ''),
                                item.get('confidence', 0),
                                item.get('raw_data', '{}'),
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                        )
                        total_new_entries += 1
                        
                        self._update_taxonomy_within_transaction(cursor, item)
                
                conn.commit()
        
        logger.info(f"Added {total_new_entries} new threat entries to database")
        
        return total_new_entries
    
    def _fetch_feed_data(self, source_name, feed_urls):
        data = []
        for feed_url in feed_urls:
            try:
                headers = {
                    'User-Agent': 'EchelonThreatIntelligence/1.0'
                }
                
                feed = feedparser.parse(feed_url, request_headers=headers)
                
                if hasattr(feed, 'entries'):
                    for entry in feed.entries:
                        processed_entry = self._process_feed_entry(entry, source_name, feed_url)
                        if processed_entry:
                            data.append(processed_entry)
            except Exception as e:
                logger.error(f"Error fetching {feed_url}: {str(e)}")
        
        return data
    
    def _process_feed_entry(self, entry, source_name, feed_url):
        try:
            title = entry.get('title', '')
            
            description = entry.get('summary', '')
            if not description and 'content' in entry:
                for content in entry.content:
                    description += content.value
            
            if not description and hasattr(entry, 'description'):
                description = entry.description
            
            description = re.sub(r'<[^>]+>', ' ', description)
            published_date = entry.get('published', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            try:
                published_date = datetime.strptime(published_date, "%a, %d %b %Y %H:%M:%S %z").strftime("%Y-%m-%d %H:%M:%S")
            except:
                try:
                    published_date = datetime.strptime(published_date, "%Y-%m-%dT%H:%M:%S%z").strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass
            
            link = entry.get('link', '')
            tags = []
            if hasattr(entry, 'tags'):
                tags = [tag.term for tag in entry.tags]
            elif hasattr(entry, 'categories'):
                tags = entry.categories
            
            text = title + " " + description
            entities = self._extract_entities(text)
            attack_vectors = self._extract_attack_vectors(description)
            sectors_affected = self._extract_sectors(text)
            regions_affected = self._extract_regions(text)

            data = {
                'title': title,
                'description': description,
                'published_date': published_date,
                'source': f"{source_name}-{feed_url}",
                'url': link,
                'tags': tags,
                'indicators': entities,
                'attack_vector': attack_vectors[0] if attack_vectors else '',
                'regions_affected': regions_affected,
                'sectors_affected': sectors_affected,
                'severity': self._determine_severity(description, entities),
                'confidence': self._calculate_confidence(description, entities),
                'raw_data': json.dumps(entry, default=str)
            }
            
            data['id'] = hashlib.sha256(f"{data['title']}-{data['published_date']}-{data['source']}".encode()).hexdigest()
            
            return data
        except Exception as e:
            logger.error(f"Error processing entry: {str(e)}")
            return None
    
    def _extract_entities(self, text):
        entities = {
            'ips': list(set(PATTERN_IP.findall(text))),
            'urls': list(set(PATTERN_URL.findall(text))),
            'cves': list(set(PATTERN_CVE.findall(text))),
            'hashes': list(set(PATTERN_HASH.findall(text)))
        }
        return entities
    
    def _extract_attack_vectors(self, text):
        attack_vectors = []
        vector_patterns = [
            (r'\b(?:spear[-\s]?phishing|phishing)\b', 'Phishing'),
            (r'\bmalware\b', 'Malware'),
            (r'\bransomware\b', 'Ransomware'),
            (r'\bddos\b', 'DDoS'),
            (r'\bsql\s+injection\b', 'SQL Injection'),
            (r'\bxss\b', 'Cross-Site Scripting'),
            (r'\bsupply\s+chain\b', 'Supply Chain Attack'),
            (r'\bzero[-\s]?day\b', 'Zero-day'),
            (r'\b(?:c2|command\s+and\s+control)\b', 'Command and Control'),
            (r'\b(?:mitm|man[-\s]in[-\s]the[-\s]middle)\b', 'Man-in-the-Middle')
        ]
        
        for pattern, vector in vector_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                attack_vectors.append(vector)
        
        return attack_vectors
    
    def _extract_sectors(self, text):
        sectors = []
        sector_patterns = [
            (r'\bfinanc(?:e|ial)\b', 'Finance'),
            (r'\bbank(?:ing)?\b', 'Finance'),
            (r'\bhealthcare\b', 'Healthcare'),
            (r'\bmedical\b', 'Healthcare'),
            (r'\bhospital\b', 'Healthcare'),
            (r'\bgovernment\b', 'Government'),
            (r'\benergy\b', 'Energy'),
            (r'\butility\b', 'Energy'),
            (r'\btech(?:nology)?\b', 'Technology')
        ]
        
        for pattern, sector in sector_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                sectors.append(sector)
        
        return list(set(sectors))
    
    def _extract_regions(self, text):
        regions = []
        region_patterns = [
            (r'\b(?:north\s+america|united\s+states|canada|mexico|usa|u\.s\.a\.|u\.s\.)\b', 'North America'),
            (r'\b(?:europe|european\s+union|eu|uk|united\s+kingdom|germany|france|italy|spain)\b', 'Europe'),
            (r'\b(?:asia|china|japan|south\s+korea|india|thailand|vietnam|indonesia)\b', 'Asia'),
            (r'\b(?:middle\s+east|iran|iraq|saudi\s+arabia|uae|israel|turkey)\b', 'Middle East')
        ]
        
        for pattern, region in region_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                regions.append(region)
        
        return list(set(regions))
    
    def _determine_severity(self, description, indicators):
        high_severity_words = ['critical', 'severe', 'high', 'urgent', 'emergency']
        medium_severity_words = ['important', 'moderate', 'medium', 'attention']
        low_severity_words = ['low', 'minor', 'limited', 'small', 'minimal']
        
        high_count = sum(1 for word in high_severity_words if re.search(r'\b' + word + r'\b', description, re.IGNORECASE))
        medium_count = sum(1 for word in medium_severity_words if re.search(r'\b' + word + r'\b', description, re.IGNORECASE))
        low_count = sum(1 for word in low_severity_words if re.search(r'\b' + word + r'\b', description, re.IGNORECASE))
        
        cve_count = len(indicators.get('cves', []))
        if cve_count > 2:
            high_count += 2
        elif cve_count > 0:
            medium_count += 1
            
        if high_count > medium_count and high_count > low_count:
            return 'High'
        elif medium_count > low_count:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_confidence(self, description, indicators):
        score = 50  
        if indicators.get('cves', []):
            score += min(20, len(indicators['cves']) * 5)  
        
        if indicators.get('ips', []):
            score += min(10, len(indicators['ips']) * 2)  
        
        if indicators.get('urls', []):
            score += min(10, len(indicators['urls']) * 2)  
        
        if indicators.get('hashes', []):
            score += min(15, len(indicators['hashes']) * 3)  

        desc_len = len(description)
        if desc_len > 1000:
            score += 10
        elif desc_len > 500:
            score += 5
        elif desc_len < 100:
            score -= 10
        
        return min(100, max(0, score))
    
    def _update_taxonomy_within_transaction(self, cursor, threat_data):
        if threat_data.get('attack_vector'):
            attack_vector = threat_data['attack_vector']
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
                    threat_data.get('source', ''),
                    json.dumps({})
                )
            )
        
        for region in threat_data.get('regions_affected', []):
            region_id = hashlib.sha256(f"region-{region}".encode()).hexdigest()
            
            cursor.execute(
                """
                INSERT OR IGNORE INTO taxonomy 
                (id, type, value, aliases, description, first_seen, last_updated, source, meta_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    region_id,
                    'region',
                    region,
                    json.dumps([]),
                    '',
                    datetime.now().strftime("%Y-%m-%d"),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    threat_data.get('source', ''),
                    json.dumps({})
                )
            )
        
        for sector in threat_data.get('sectors_affected', []):
            sector_id = hashlib.sha256(f"sector-{sector}".encode()).hexdigest()
            
            cursor.execute(
                """
                INSERT OR IGNORE INTO taxonomy 
                (id, type, value, aliases, description, first_seen, last_updated, source, meta_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sector_id,
                    'sector',
                    sector,
                    json.dumps([]),
                    '',
                    datetime.now().strftime("%Y-%m-%d"),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    threat_data.get('source', ''),
                    json.dumps({})
                )
            )
    
    def setup_scheduled_collection(self):
        def run_collection():
            try:
                logger.info("Running scheduled data collection")
                self.collect_data()
            except Exception as e:
                logger.error(f"Error in scheduled collection: {str(e)}")
        
        schedule.every(DATA_COLLECTION_INTERVAL_HOURS).hours.do(run_collection)
        
        thread = threading.Thread(target=self._run_scheduler, daemon=True)
        thread.start()
    
    def _run_scheduler(self):
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def get_recent_threats(self, days=30, limit=100):
        date_threshold = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        with self.db_lock:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    """
                    SELECT * FROM threats 
                    WHERE published_date >= ?
                    ORDER BY published_date DESC
                    LIMIT ?
                    """, 
                    (date_threshold, limit)
                )
                
                threats = cursor.fetchall()
        
        for threat in threats:
            threat['tags'] = json.loads(threat['tags'])
            threat['indicators'] = json.loads(threat['indicators'])
            threat['regions_affected'] = json.loads(threat['regions_affected'])
            threat['sectors_affected'] = json.loads(threat['sectors_affected'])
        
        return threats
    
    def get_threat_by_id(self, threat_id):
        with self.db_lock:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM threats WHERE id = ?", (threat_id,))
                threat = cursor.fetchone()
        
        if not threat:
            return None
        
        threat['tags'] = json.loads(threat['tags'])
        threat['indicators'] = json.loads(threat['indicators'])
        threat['regions_affected'] = json.loads(threat['regions_affected'])
        threat['sectors_affected'] = json.loads(threat['sectors_affected'])
        
        return threat
    
    def get_taxonomy_values(self, type_name):
        with self.db_lock:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT value FROM taxonomy WHERE type = ?", (type_name,))
                values = [row['value'] for row in cursor.fetchall()]
        
        return values
    
    def store_prediction(self, prediction):
        if 'id' not in prediction:
            prediction['id'] = hashlib.sha256(f"{prediction['apt_group']}-{prediction['timestamp']}".encode()).hexdigest()
        
        if 'created_at' not in prediction:
            prediction['created_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with self.db_lock:
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
                        prediction['apt_group'],
                        prediction['attack_type'],
                        prediction['threat_category'],
                        prediction['region'],
                        prediction['industry'],
                        prediction['severity'],
                        prediction['likelihood'],
                        prediction['confidence'],
                        prediction['timestamp'],
                        prediction['description'],
                        json.dumps(prediction.get('indicators', {})),
                        prediction['affecting'],
                        json.dumps(prediction.get('evidence', [])),
                        prediction['created_at']
                    )
                )
                
                conn.commit()
    
    def get_recent_predictions(self, limit=20):
        with self.db_lock:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    """
                    SELECT * FROM predictions 
                    ORDER BY timestamp ASC
                    LIMIT ?
                    """, 
                    (limit,)
                )
                
                predictions = cursor.fetchall()
        
        for prediction in predictions:
            prediction['indicators'] = json.loads(prediction['indicators'])
            prediction['evidence'] = json.loads(prediction['evidence'])
        
        return predictions
