#!/bin/bash
# enhance_predictive_capabilities.sh - Transform Echelon to predict future cyber attacks using only real data

echo "========================================="
echo "ECHELON: ENHANCING PREDICTIVE CAPABILITIES"
echo "========================================="
echo "Installing advanced predictive capabilities for future attack forecasting"
echo "USING ONLY REAL DATA - NO SIMULATIONS"

# Create necessary directories
mkdir -p models/predictive
mkdir -p data/trends
mkdir -p data/raw/threat_feeds
mkdir -p data/processed/attack_forecasts
mkdir -p scripts/predictive

# Step 1: Install required Python dependencies for advanced analysis
echo "Installing advanced data science and ML dependencies..."
pip install pandas scikit-learn numpy tensorflow keras prophet nltk spacy gensim transformers statsmodels plotly tqdm joblib datasets requests-html

# Step 2: Create real-data collector for additional sources
echo "Creating enhanced real data collector..."
cat > scripts/predictive/real_data_collector.py << 'EOT'
#!/usr/bin/env python3
import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import re
import feedparser
import threading
import logging
from requests_html import HTMLSession
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/trends/collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DataCollector')

class RealDataCollector:
    """Collects only real threat intelligence data from legitimate sources"""
    
    def __init__(self, api_keys_path='config/api_keys.json'):
        self.api_keys = self.load_api_keys(api_keys_path)
        self.headers = {
            'User-Agent': 'Echelon Threat Intelligence System/1.0'
        }
        self.raw_data_dir = 'data/raw/threat_feeds'
        self.trends_dir = 'data/trends'
        
        # Create directories
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.trends_dir, exist_ok=True)
    
    def load_api_keys(self, path):
        """Load API keys from configuration"""
        if not os.path.exists(path):
            logger.error(f"API keys file not found at {path}")
            return {}
        
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
            return {}
    
    def collect_all_data(self):
        """Collect data from all available sources"""
        logger.info("Starting comprehensive real data collection...")
        
        # Create status tracker for collection
        collection_status = {
            "start_time": datetime.now().isoformat(),
            "sources": {}
        }
        
        # Collect from each source, one at a time to ensure we get proper error handling
        sources = {
            "otx_pulses": self.collect_otx_pulses,
            "mitre_attack": self.collect_mitre_data,
            "nvd_vulnerabilities": self.collect_nvd,
            "cisa_kev": self.collect_cisa_kev,
            "feodotracker": self.collect_feodotracker,
            "urlhaus": self.collect_urlhaus,
            "security_rss": self.collect_security_rss,
            "threat_research": self.collect_threat_research,
            "vt_livehunts": self.collect_virustotal,
        }
        
        for source_name, collection_function in sources.items():
            try:
                logger.info(f"Collecting from {source_name}...")
                result = collection_function()
                collection_status["sources"][source_name] = {
                    "status": "success",
                    "items_collected": result,
                    "timestamp": datetime.now().isoformat()
                }
                logger.info(f"✓ Collected {result} items from {source_name}")
            except Exception as e:
                logger.error(f"✗ Error collecting from {source_name}: {e}")
                collection_status["sources"][source_name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        # Save collection status
        collection_status["end_time"] = datetime.now().isoformat()
        with open(os.path.join(self.trends_dir, f"collection_status_{datetime.now().strftime('%Y%m%d')}.json"), 'w') as f:
            json.dump(collection_status, f, indent=2)
        
        logger.info("All real data collection complete!")
        
        return collection_status
    
    def collect_otx_pulses(self):
        """Collect real threat intelligence from AlienVault OTX"""
        if 'alienvault_otx' not in self.api_keys or not self.api_keys.get('alienvault_otx', {}).get('api_key'):
            logger.error("No AlienVault OTX API key configured")
            return 0
        
        api_key = self.api_keys['alienvault_otx']['api_key']
        headers = self.headers.copy()
        headers['X-OTX-API-KEY'] = api_key
        
        url = 'https://otx.alienvault.com/api/v1/pulses/subscribed'
        params = {'limit': 50}  # Get the 50 most recent pulses
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Save raw data
            output_path = os.path.join(self.raw_data_dir, f"otx_pulses_{datetime.now().strftime('%Y%m%d')}.json")
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            pulses = data.get('results', [])
            
            # Optionally fetch indicators for each pulse
            pulses_with_indicators = []
            for pulse in pulses[:10]:  # Limit to 10 pulses to avoid rate limiting
                pulse_id = pulse.get('id')
                if pulse_id:
                    indicators_url = f'https://otx.alienvault.com/api/v1/pulses/{pulse_id}/indicators'
                    try:
                        indicator_response = requests.get(indicators_url, headers=headers, timeout=30)
                        indicator_response.raise_for_status()
                        
                        indicator_data = indicator_response.json()
                        pulse['indicators'] = indicator_data.get('results', [])
                        
                        # Save indicators for this pulse
                        indicator_path = os.path.join(self.raw_data_dir, f"otx_pulse_{pulse_id}_indicators.json")
                        with open(indicator_path, 'w') as f:
                            json.dump(indicator_data, f, indent=2)
                        
                        pulses_with_indicators.append(pulse)
                        time.sleep(1)  # Be nice to the API
                    except Exception as e:
                        logger.error(f"Error fetching indicators for pulse {pulse_id}: {e}")
            
            # Save processed data with indicators
            processed_path = os.path.join(self.trends_dir, f"otx_processed_{datetime.now().strftime('%Y%m%d')}.json")
            with open(processed_path, 'w') as f:
                json.dump(pulses_with_indicators, f, indent=2)
            
            return len(pulses)
            
        except Exception as e:
            logger.error(f"Error collecting OTX pulses: {e}")
            raise
    
    def collect_mitre_data(self):
        """Collect comprehensive MITRE ATT&CK framework data"""
        mitre_urls = [
            'https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json',
            'https://raw.githubusercontent.com/mitre/cti/master/mobile-attack/mobile-attack.json',
            'https://raw.githubusercontent.com/mitre/cti/master/ics-attack/ics-attack.json'
        ]
        
        all_objects = []
        
        for url in mitre_urls:
            try:
                # Extract dataset name
                dataset_name = url.split('/')[-1].split('.')[0]
                
                logger.info(f"Fetching MITRE dataset: {dataset_name}")
                
                # Make request
                response = requests.get(url, headers=self.headers, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                
                # Save raw data
                output_path = os.path.join(self.raw_data_dir, f"mitre_{dataset_name}.json")
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Extract objects
                objects = data.get('objects', [])
                logger.info(f"Retrieved {len(objects)} objects from {dataset_name}")
                
                all_objects.extend(objects)
                
                # Add delay between requests
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching MITRE data from {url}: {e}")
        
        # Process MITRE data into attack patterns, techniques, and relationships
        if all_objects:
            try:
                # Extract attack patterns (techniques)
                attack_patterns = [obj for obj in all_objects if obj.get('type') == 'attack-pattern']
                
                # Extract relationships
                relationships = [obj for obj in all_objects if obj.get('type') == 'relationship']
                
                # Extract campaigns
                campaigns = [obj for obj in all_objects if obj.get('type') == 'campaign']
                
                # Extract intrusion sets (threat actors/groups)
                intrusion_sets = [obj for obj in all_objects if obj.get('type') == 'intrusion-set']
                
                # Extract tools and malware
                tools = [obj for obj in all_objects if obj.get('type') == 'tool']
                malware = [obj for obj in all_objects if obj.get('type') == 'malware']
                
                # Create organized structure for the prediction model
                mitre_structure = {
                    "attack_patterns": attack_patterns,
                    "relationships": relationships,
                    "campaigns": campaigns,
                    "intrusion_sets": intrusion_sets,
                    "tools": tools,
                    "malware": malware,
                    "meta": {
                        "collected_at": datetime.now().isoformat(),
                        "total_objects": len(all_objects)
                    }
                }
                
                # Save processed structure
                structure_path = os.path.join(self.trends_dir, "mitre_structure.json")
                with open(structure_path, 'w') as f:
                    json.dump(mitre_structure, f, indent=2)
                
                logger.info(f"Processed MITRE data: {len(attack_patterns)} techniques, {len(intrusion_sets)} threat actors")
            
            except Exception as e:
                logger.error(f"Error processing MITRE data: {e}")
        
        return len(all_objects)
    
    def collect_nvd(self):
        """Collect vulnerability data from NVD"""
        # NVD API URL
        base_url = 'https://services.nvd.nist.gov/rest/json/cves/2.0'
        
        # Get vulnerabilities from last 30 days
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%dT00:00:00.000')
        
        params = {
            'pubStartDate': thirty_days_ago,
            'resultsPerPage': 100
        }
        
        all_vulns = []
        
        try:
            # Make initial request
            response = requests.get(base_url, headers=self.headers, params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            # Save raw data
            output_path = os.path.join(self.raw_data_dir, f"nvd_vulns_{datetime.now().strftime('%Y%m%d')}.json")
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Extract vulnerabilities
            vulns = data.get('vulnerabilities', [])
            all_vulns.extend(vulns)
            
            # Check if there are more pages
            total_results = data.get('totalResults', 0)
            results_per_page = data.get('resultsPerPage', 0)
            
            if total_results > results_per_page:
                # Calculate number of pages
                num_pages = (total_results + results_per_page - 1) // results_per_page
                
                # Fetch remaining pages
                for page in range(1, min(num_pages, 5)):  # Limit to 5 pages to avoid overloading
                    params['startIndex'] = page * results_per_page
                    
                    logger.info(f"Fetching NVD page {page+1}/{num_pages}")
                    
                    response = requests.get(base_url, headers=self.headers, params=params, timeout=60)
                    response.raise_for_status()
                    
                    page_data = response.json()
                    page_vulns = page_data.get('vulnerabilities', [])
                    
                    # Save page data
                    page_path = os.path.join(self.raw_data_dir, f"nvd_vulns_page{page+1}_{datetime.now().strftime('%Y%m%d')}.json")
                    with open(page_path, 'w') as f:
                        json.dump(page_data, f, indent=2)
                    
                    all_vulns.extend(page_vulns)
                    
                    # Add delay
                    time.sleep(6)  # NVD rate limit is 5 requests per 30 seconds
            
            # Process vulnerabilities into a more usable format
            processed_vulns = []
            for vuln in all_vulns:
                cve = vuln.get('cve', {})
                cve_id = cve.get('id', '')
                
                if cve_id:
                    # Extract description
                    description = ""
                    if cve.get('descriptions'):
                        for desc in cve.get('descriptions', []):
                            if desc.get('lang') == 'en':
                                description = desc.get('value', '')
                                break
                    
                    # Extract CVSS data
                    metrics = cve.get('metrics', {})
                    cvss_v3 = None
                    severity = "UNKNOWN"
                    base_score = 0
                    
                    # Try CVSS v3.1
                    if metrics.get('cvssMetricV31'):
                        cvss_v3 = metrics.get('cvssMetricV31')[0].get('cvssData', {})
                        severity = cvss_v3.get('baseSeverity', 'UNKNOWN')
                        base_score = float(cvss_v3.get('baseScore', 0))
                    # Try CVSS v3.0
                    elif metrics.get('cvssMetricV30'):
                        cvss_v3 = metrics.get('cvssMetricV30')[0].get('cvssData', {})
                        severity = cvss_v3.get('baseSeverity', 'UNKNOWN')
                        base_score = float(cvss_v3.get('baseScore', 0))
                    
                    # Create simplified CVE object
                    processed_vuln = {
                        "cve_id": cve_id,
                        "description": description,
                        "published": cve.get('published', ''),
                        "last_modified": cve.get('lastModified', ''),
                        "severity": severity,
                        "base_score": base_score,
                        "source": "NVD"
                    }
                    
                    processed_vulns.append(processed_vuln)
            
            # Save processed vulnerabilities
            processed_path = os.path.join(self.trends_dir, f"nvd_processed_{datetime.now().strftime('%Y%m%d')}.json")
            with open(processed_path, 'w') as f:
                json.dump(processed_vulns, f, indent=2)
            
            return len(processed_vulns)
            
        except Exception as e:
            logger.error(f"Error collecting NVD data: {e}")
            raise
    
    def collect_cisa_kev(self):
        """Collect CISA Known Exploited Vulnerabilities catalog"""
        url = 'https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv'
        
        try:
            # Make request
            response = requests.get(url, headers=self.headers, timeout=60)
            response.raise_for_status()
            
            # Save raw CSV
            csv_path = os.path.join(self.raw_data_dir, f"cisa_kev_{datetime.now().strftime('%Y%m%d')}.csv")
            with open(csv_path, 'w') as f:
                f.write(response.text)
            
            # Parse CSV into usable format
            try:
                # Use pandas to parse CSV
                df = pd.read_csv(csv_path)
                
                # Convert to list of dictionaries
                kev_list = df.to_dict('records')
                
                # Save as JSON for easier processing
                json_path = os.path.join(self.trends_dir, f"cisa_kev_{datetime.now().strftime('%Y%m%d')}.json")
                with open(json_path, 'w') as f:
                    json.dump(kev_list, f, indent=2)
                
                logger.info(f"Processed {len(kev_list)} KEV entries")
                return len(kev_list)
                
            except Exception as e:
                logger.error(f"Error processing CISA KEV CSV: {e}")
                # Try manual parsing if pandas fails
                lines = response.text.strip().split('\n')
                if len(lines) > 1:
                    return len(lines) - 1  # Subtract header row
                return 0
                
        except Exception as e:
            logger.error(f"Error fetching CISA KEV: {e}")
            raise
    
    def collect_feodotracker(self):
        """Collect data from Feodo Tracker (Emotet/Dridex/TrickBot/QakBot C2 servers)"""
        url = 'https://feodotracker.abuse.ch/downloads/ipblocklist.json'
        
        try:
            # Make request
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Save raw data
            output_path = os.path.join(self.raw_data_dir, f"feodotracker_{datetime.now().strftime('%Y%m%d')}.json")
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return len(data)
            
        except Exception as e:
            logger.error(f"Error collecting Feodo Tracker data: {e}")
            raise
    
    def collect_urlhaus(self):
        """Collect data from URLhaus (malicious URLs)"""
        url = 'https://urlhaus.abuse.ch/downloads/json/'
        
        try:
            # Make request
            response = requests.get(url, headers=self.headers, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            # Save raw data
            output_path = os.path.join(self.raw_data_dir, f"urlhaus_{datetime.now().strftime('%Y%m%d')}.json")
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Extract useful data for prediction
            urls = data.get('urls', [])
            
            # Process into more usable format
            processed_urls = []
            for url_entry in urls:
                processed_urls.append({
                    "url": url_entry.get('url', ''),
                    "date_added": url_entry.get('date_added', ''),
                    "threat": url_entry.get('threat', ''),
                    "tags": url_entry.get('tags', []),
                    "malware_type": url_entry.get('malware', '')
                })
            
            # Save processed data
            processed_path = os.path.join(self.trends_dir, f"urlhaus_processed_{datetime.now().strftime('%Y%m%d')}.json")
            with open(processed_path, 'w') as f:
                json.dump(processed_urls, f, indent=2)
            
            return len(urls)
            
        except Exception as e:
            logger.error(f"Error collecting URLhaus data: {e}")
            raise
    
    def collect_security_rss(self):
        """Collect security news from various RSS feeds"""
        security_feeds = [
            'https://www.wired.com/feed/category/security/latest/rss',
            'https://krebsonsecurity.com/feed/',
            'https://www.schneier.com/feed/atom/',
            'https://www.bleepingcomputer.com/feed/',
            'https://threatpost.com/feed/',
            'https://www.securityweek.com/feed/'
        ]
        
        all_articles = []
        
        for feed_url in security_feeds:
            try:
                # Extract feed name
                feed_name = feed_url.split('/')[2]
                
                logger.info(f"Fetching security feed: {feed_name}")
                
                # Parse RSS feed
                feed = feedparser.parse(feed_url)
                
                # Extract articles
                articles = []
                for entry in feed.entries:
                    article = {
                        "title": entry.get('title', ''),
                        "link": entry.get('link', ''),
                        "published": entry.get('published', ''),
                        "summary": entry.get('summary', ''),
                        "source": feed_name
                    }
                    articles.append(article)
                
                # Save feed data
                feed_path = os.path.join(self.raw_data_dir, f"security_feed_{feed_name}_{datetime.now().strftime('%Y%m%d')}.json")
                with open(feed_path, 'w') as f:
                    json.dump(articles, f, indent=2)
                
                all_articles.extend(articles)
                
                # Add delay
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching security feed {feed_url}: {e}")
        
        # Save combined articles
        combined_path = os.path.join(self.trends_dir, f"security_news_{datetime.now().strftime('%Y%m%d')}.json")
        with open(combined_path, 'w') as f:
            json.dump(all_articles, f, indent=2)
        
        return len(all_articles)
    
    def collect_threat_research(self):
        """Collect threat research from security vendors using web scraping"""
        # Use requests-html for simple scraping
        session = HTMLSession()
        
        research_sources = [
            {"name": "mandiant", "url": "https://www.mandiant.com/resources/blog"},
            {"name": "crowdstrike", "url": "https://www.crowdstrike.com/blog/category/threat-intel-research/"},
            {"name": "talos", "url": "https://blog.talosintelligence.com/"}
        ]
        
        all_reports = []
        
        for source in research_sources:
            try:
                logger.info(f"Fetching threat research from {source['name']}")
                
                # Make request
                response = session.get(source['url'])
                response.raise_for_status()
                
                # Render JavaScript if needed
                response.html.render(timeout=30, sleep=1)
                
                # Extract article links and titles
                articles = []
                
                # Different selectors based on source
                if source['name'] == 'mandiant':
                    article_elements = response.html.find('.blog-card')
                    for element in article_elements:
                        title_elem = element.find('.blog-title', first=True)
                        link_elem = element.find('a', first=True)
                        
                        if title_elem and link_elem:
                            articles.append({
                                "title": title_elem.text,
                                "url": link_elem.attrs.get('href', ''),
                                "source": source['name']
                            })
                
                elif source['name'] == 'crowdstrike':
                    article_elements = response.html.find('.post')
                    for element in article_elements:
                        title_elem = element.find('.entry-title a', first=True)
                        
                        if title_elem:
                            articles.append({
                                "title": title_elem.text,
                                "url": title_elem.attrs.get('href', ''),
                                "source": source['name']
                            })
                
                elif source['name'] == 'talos':
                    article_elements = response.html.find('.post')
                    for element in article_elements:
                        title_elem = element.find('.post-title a', first=True)
                        
                        if title_elem:
                            articles.append({
                                "title": title_elem.text,
                                "url": title_elem.attrs.get('href', ''),
                                "source": source['name']
                            })
                
                # Save source data
                source_path = os.path.join(self.raw_data_dir, f"threat_research_{source['name']}_{datetime.now().strftime('%Y%m%d')}.json")
                with open(source_path, 'w') as f:
                    json.dump(articles, f, indent=2)
                
                all_reports.extend(articles)
                
                # Add delay
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error fetching threat research from {source['name']}: {e}")
        
        # Save combined reports
        combined_path = os.path.join(self.trends_dir, f"threat_research_{datetime.now().strftime('%Y%m%d')}.json")
        with open(combined_path, 'w') as f:
            json.dump(all_reports, f, indent=2)
        
        return len(all_reports)
    
    def collect_virustotal(self):
        """Collect threat intelligence from VirusTotal LiveHunt if API key available"""
        if 'virustotal' not in self.api_keys or not self.api_keys.get('virustotal', {}).get('api_key'):
            logger.info("No VirusTotal API key configured, skipping")
            return 0
        
        api_key = self.api_keys['virustotal']['api_key']
        headers = self.headers.copy()
        headers['x-apikey'] = api_key
        
        url = 'https://www.virustotal.com/api/v3/intelligence/hunting_notification_files'
        
        try:
            # Make request
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            
            # Save raw data
            output_path = os.path.join(self.raw_data_dir, f"virustotal_livehunt_{datetime.now().strftime('%Y%m%d')}.json")
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Extract file notifications
            notifications = data.get('data', [])
            
            return len(notifications)
            
        except Exception as e:
            logger.error(f"Error collecting VirusTotal data: {e}")
            raise

# Main execution
if __name__ == "__main__":
    collector = RealDataCollector()
    collector.collect_all_data()
EOT

chmod +x scripts/predictive/real_data_collector.py

# Step 3: Create advanced predictive model using real data
echo "Creating advanced real-data predictive model..."
cat > models/predictive/future_attack_predictor.py << 'EOT'
#!/usr/bin/env python3
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from prophet import Prophet
import logging
import re
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('models/predictive/predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AttackPredictor')

class FutureAttackPredictor:
    """Predicts future cyber attacks using real threat intelligence data"""
    
    def __init__(self):
        self.raw_data_dir = 'data/raw/threat_feeds'
        self.trends_dir = 'data/trends'
        self.models_dir = 'models/predictive'
        self.forecasts_dir = 'data/processed/attack_forecasts'
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.forecasts_dir, exist_ok=True)
        
        # Initialize models
        self.time_series_model = None
        self.attack_pattern_model = None
        self.target_prediction_model = None
        
        # Load models if they exist
        self.load_models()
    
    def load_models(self):
        """Load trained models if available"""
        try:
            # Time series model (Prophet)
            if os.path.exists(os.path.join(self.models_dir, 'time_series_model.pkl')):
                with open(os.path.join(self.models_dir, 'time_series_model.pkl'), 'rb') as f:
                    self.time_series_model = pickle.load(f)
                logger.info("Loaded time series model")
            
            # Attack pattern model (Keras)
            if os.path.exists(os.path.join(self.models_dir, 'attack_pattern_model.h5')):
                self.attack_pattern_model = keras.models.load_model(os.path.join(self.models_dir, 'attack_pattern_model.h5'))
                logger.info("Loaded attack pattern model")
            
            # Target prediction model
            if os.path.exists(os.path.join(self.models_dir, 'target_prediction_model.pkl')):
                with open(os.path.join(self.models_dir, 'target_prediction_model.pkl'), 'rb') as f:
                    self.target_prediction_model = pickle.load(f)
                logger.info("Loaded target prediction model")
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def collect_all_data(self):
        """Collect all available real data for training"""
        data = {
            'vulnerabilities': [],
            'mitre_techniques': [],
            'mitre_groups': [],
            'threat_reports': [],
            'indicators': [],
            'security_news': []
        }
        
        # Collect vulnerabilities from NVD and CISA KEV
        data['vulnerabilities'].extend(self.collect_vulnerabilities())
        
        # Collect MITRE ATT&CK data
        mitre_data = self.collect_mitre_data()
        data['mitre_techniques'] = mitre_data.get('techniques', [])
        data['mitre_groups'] = mitre_data.get('groups', [])
        
        # Collect threat intelligence
        data['indicators'].extend(self.collect_threat_intel())
        
        # Collect security news and reports
        data['security_news'].extend(self.collect_security_news())
        data['threat_reports'].extend(self.collect_threat_reports())
        
        logger.info(f"Collected {sum(len(v) for v in data.values())} total data points")
        
        return data
    
    def collect_vulnerabilities(self):
        """Collect vulnerability data from NVD and CISA KEV"""
        vulnerabilities = []
        
        # Look for NVD data
        nvd_files = glob.glob(os.path.join(self.trends_dir, 'nvd_processed_*.json'))
        if nvd_files:
            latest_nvd = max(nvd_files)
            try:
                with open(latest_nvd, 'r') as f:
                    nvd_vulns = json.load(f)
                    logger.info(f"Loaded {len(nvd_vulns)} vulnerabilities from NVD")
                    vulnerabilities.extend(nvd_vulns)
            except Exception as e:
                logger.error(f"Error loading NVD data: {e}")
        
        # Look for CISA KEV data
        kev_files = glob.glob(os.path.join(self.trends_dir, 'cisa_kev_*.json'))
        if kev_files:
            latest_kev = max(kev_files)
            try:
                with open(latest_kev, 'r') as f:
                    kev_vulns = json.load(f)
                    # Mark as known exploited
                    for vuln in kev_vulns:
                        vuln['known_exploited'] = True
                    logger.info(f"Loaded {len(kev_vulns)} vulnerabilities from CISA KEV")
                    vulnerabilities.extend(kev_vulns)
            except Exception as e:
                logger.error(f"Error loading CISA KEV data: {e}")
        
        return vulnerabilities
    
    def collect_mitre_data(self):
        """Collect MITRE ATT&CK framework data"""
        result = {
            'techniques': [],
            'groups': [],
            'software': [],
            'relationships': []
        }
        
        # Check if we have processed MITRE data
        if os.path.exists(os.path.join(self.trends_dir, 'mitre_structure.json')):
            try:
                with open(os.path.join(self.trends_dir, 'mitre_structure.json'), 'r') as f:
                    mitre_data = json.load(f)
                    
                    # Extract techniques
                    techniques = mitre_data.get('attack_patterns', [])
                    result['techniques'] = techniques
                    
                    # Extract groups
                    groups = mitre_data.get('intrusion_sets', [])
                    result['groups'] = groups
                    
                    # Extract software (tools and malware)
                    software = mitre_data.get('tools', []) + mitre_data.get('malware', [])
                    result['software'] = software
                    
                    # Extract relationships
                    relationships = mitre_data.get('relationships', [])
                    result['relationships'] = relationships
                    
                    logger.info(f"Loaded MITRE data: {len(techniques)} techniques, {len(groups)} groups")
            except Exception as e:
                logger.error(f"Error loading MITRE data: {e}")
        
        return result
    
    def collect_threat_intel(self):
        """Collect threat intelligence indicators"""
        indicators = []
        
        # Look for OTX data
        otx_files = glob.glob(os.path.join(self.trends_dir, 'otx_processed_*.json'))
        if otx_files:
            latest_otx = max(otx_files)
            try:
                with open(latest_otx, 'r') as f:
                    otx_data = json.load(f)
                    # Extract indicators
                    for pulse in otx_data:
                        if 'indicators' in pulse:
                            pulse_indicators = pulse.get('indicators', [])
                            # Add pulse context to each indicator
                            for indicator in pulse_indicators:
                                indicator['pulse_name'] = pulse.get('name', '')
                                indicator['pulse_tags'] = pulse.get('tags', [])
                                indicator['pulse_created'] = pulse.get('created', '')
                                indicator['source'] = 'AlienVault OTX'
                            indicators.extend(pulse_indicators)
                    logger.info(f"Loaded {len(indicators)} indicators from OTX")
            except Exception as e:
                logger.error(f"Error loading OTX data: {e}")
        
        # Look for URLhaus data
        urlhaus_files = glob.glob(os.path.join(self.trends_dir, 'urlhaus_processed_*.json'))
        if urlhaus_files:
            latest_urlhaus = max(urlhaus_files)
            try:
                with open(latest_urlhaus, 'r') as f:
                    urlhaus_data = json.load(f)
                    # Convert to indicator format
                    urlhaus_indicators = []
                    for url_entry in urlhaus_data:
                        indicator = {
                            'type': 'url',
                            'indicator': url_entry.get('url', ''),
                            'created': url_entry.get('date_added', ''),
                            'threat': url_entry.get('threat', ''),
                            'tags': url_entry.get('tags', []),
                            'source': 'URLhaus'
                        }
                        urlhaus_indicators.append(indicator)
                    indicators.extend(urlhaus_indicators)
                    logger.info(f"Loaded {len(urlhaus_indicators)} indicators from URLhaus")
            except Exception as e:
                logger.error(f"Error loading URLhaus data: {e}")
        
        # Look for Feodotracker data
        feodo_files = glob.glob(os.path.join(self.raw_data_dir, 'feodotracker_*.json'))
        if feodo_files:
            latest_feodo = max(feodo_files)
            try:
                with open(latest_feodo, 'r') as f:
                    feodo_data = json.load(f)
                    # Convert to indicator format
                    feodo_indicators = []
                    for entry in feodo_data:
                        indicator = {
                            'type': 'ip',
                            'indicator': entry.get('ip_address', entry.get('ipv4', '')),
                            'created': entry.get('first_seen', ''),
                            'malware_family': entry.get('malware', ''),
                            'source': 'Feodotracker'
                        }
                        if indicator['indicator']:
                            feodo_indicators.append(indicator)
                    indicators.extend(feodo_indicators)
                    logger.info(f"Loaded {len(feodo_indicators)} indicators from Feodotracker")
            except Exception as e:
                logger.error(f"Error loading Feodotracker data: {e}")
        
        return indicators
    
    def collect_security_news(self):
        """Collect security news"""
        news_articles = []
        
        # Look for security news data
        news_files = glob.glob(os.path.join(self.trends_dir, 'security_news_*.json'))
        if news_files:
            latest_news = max(news_files)
            try:
                with open(latest_news, 'r') as f:
                    news_data = json.load(f)
                    news_articles.extend(news_data)
                    logger.info(f"Loaded {len(news_data)} security news articles")
            except Exception as e:
                logger.error(f"Error loading security news: {e}")
        
        return news_articles
    
    def collect_threat_reports(self):
        """Collect threat research reports"""
        reports = []
        
        # Look for threat research data
        report_files = glob.glob(os.path.join(self.trends_dir, 'threat_research_*.json'))
        if report_files:
            latest_report = max(report_files)
            try:
                with open(latest_report, 'r') as f:
                    report_data = json.load(f)
                    reports.extend(report_data)
                    logger.info(f"Loaded {len(report_data)} threat research reports")
            except Exception as e:
                logger.error(f"Error loading threat research: {e}")
        
        return reports
    
    def prepare_time_series_data(self, data):
        """Prepare time series data for Prophet model"""
        # Extract dates from various sources
        dates = []
        
        # Vulnerabilities
        for vuln in data.get('vulnerabilities', []):
            if 'published' in vuln and vuln['published']:
                try:
                    # Handle different date formats
                    date_str = vuln['published']
                    if 'T' in date_str:
                        # ISO format
                        date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    else:
                        # Try various formats
                        date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']
                        for fmt in date_formats:
                            try:
                                date = datetime.strptime(date_str, fmt)
                                break
                            except:
                                continue
                    dates.append({'ds': date, 'y': 1, 'type': 'vulnerability'})
                except:
                    pass
        
        # Indicators
        for indicator in data.get('indicators', []):
            if 'created' in indicator and indicator['created']:
                try:
                    # Handle different date formats
                    date_str = indicator['created']
                    if 'T' in date_str:
                        # ISO format
                        date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    else:
                        # Try various formats
                        date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']
                        for fmt in date_formats:
                            try:
                                date = datetime.strptime(date_str, fmt)
                                break
                            except:
                                continue
                    dates.append({'ds': date, 'y': 1, 'type': 'indicator'})
                except:
                    pass
        
        # Convert to DataFrame
        if dates:
            df = pd.DataFrame(dates)
            
            # Group by date
            daily_counts = df.groupby(['ds', 'type']).count().reset_index()
            
            # Pivot to get separate columns for each type
            pivot_df = daily_counts.pivot_table(index='ds', columns='type', values='y', fill_value=0).reset_index()
            
            # Add total column
            pivot_df['total'] = pivot_df.sum(axis=1, numeric_only=True)
            
            logger.info(f"Prepared time series data with {len(pivot_df)} data points")
            return pivot_df
        else:
            logger.warning("No valid dates found for time series analysis")
            return None
    
    def train_time_series_model(self, data):
        """Train Prophet model for time series forecasting"""
        # Prepare time series data
        ts_data = self.prepare_time_series_data(data)
        
        if ts_data is None or len(ts_data) < 7:
            logger.warning("Insufficient data for time series modeling")
            return False
        
        try:
            # Use 'total' column for forecasting
            forecast_df = ts_data[['ds', 'total']].rename(columns={'total': 'y'})
            
            # Train Prophet model
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                interval_width=0.95
            )
            
            # Add country-specific holidays if available
            try:
                model.add_country_holidays(country_name='US')
            except:
                pass
            
            # Fit model
            model.fit(forecast_df)
            
            # Save model
            with open(os.path.join(self.models_dir, 'time_series_model.pkl'), 'wb') as f:
                pickle.dump(model, f)
            
            self.time_series_model = model
            logger.info("Successfully trained time series model")
            
            return True
        except Exception as e:
            logger.error(f"Error training time series model: {e}")
            return False
    
    def train_attack_pattern_model(self, data):
        """Train model to predict attack patterns based on MITRE ATT&CK data"""
        # Extract MITRE data
        techniques = data.get('mitre_techniques', [])
        groups = data.get('mitre_groups', [])
        relationships = data.get('relationships', [])
        
        if not techniques or not groups or not relationships:
            logger.warning("Insufficient MITRE data for attack pattern modeling")
            return False
        
        try:
            # First, map techniques to groups using relationships
            group_techniques = {}
            
            for relationship in relationships:
                if relationship.get('relationship_type') == 'uses':
                    source_ref = relationship.get('source_ref', '')
                    target_ref = relationship.get('target_ref', '')
                    
                    # Check if source is a group and target is a technique
                    if source_ref.startswith('intrusion-set--') and target_ref.startswith('attack-pattern--'):
                        group_id = source_ref
                        technique_id = target_ref
                        
                        if group_id not in group_techniques:
                            group_techniques[group_id] = []
                        
                        group_techniques[group_id].append(technique_id)
            
            # Filter to groups that have multiple techniques
            filtered_groups = {group_id: techs for group_id, techs in group_techniques.items() if len(techs) >= 3}
            
            if not filtered_groups:
                logger.warning("Insufficient technique data for attack pattern modeling")
                return False
            
            # Create sequences of techniques for each group
            sequences = []
            
            for group_id, technique_ids in filtered_groups.items():
                # Get technique names and ATT&CK IDs
                technique_info = []
                for tech_id in technique_ids:
                    # Find technique details
                    for technique in techniques:
                        if technique.get('id') == tech_id:
                            # Get ATT&CK ID from external references
                            attack_id = None
                            for ref in technique.get('external_references', []):
                                if ref.get('source_name') == 'mitre-attack':
                                    attack_id = ref.get('external_id')
                                    break
                            
                            if attack_id:
                                technique_info.append({
                                    'id': attack_id,
                                    'name': technique.get('name', '')
                                })
                            break
                
                if len(technique_info) >= 3:
                    sequences.append(technique_info)
            
            if not sequences:
                logger.warning("Could not extract valid technique sequences")
                return False
            
            # Prepare data for sequential model
            # Create mapping of technique IDs to integers
            all_techniques = set()
            for seq in sequences:
                for tech in seq:
                    all_techniques.add(tech['id'])
            
            technique_mapping = {tech_id: i+1 for i, tech_id in enumerate(sorted(all_techniques))}
            
            # Save technique mapping
            with open(os.path.join(self.models_dir, 'technique_mapping.json'), 'w') as f:
                json.dump(technique_mapping, f, indent=2)
            
            # Create sequences for training
            X = []
            y = []
            
            seq_length = 3  # Use 3 techniques to predict the next one
            
            for sequence in sequences:
                # Convert to IDs
                seq_ids = [technique_mapping[tech['id']] for tech in sequence]
                
                # Create training samples
                for i in range(len(seq_ids) - seq_length):
                    X.append(seq_ids[i:i+seq_length])
                    y.append(seq_ids[i+seq_length])
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            if len(X) < 10:
                logger.warning("Insufficient training samples for attack pattern model")
                return False
            
            # One-hot encode labels
            num_techniques = len(technique_mapping) + 1  # +1 for padding
            y_onehot = tf.keras.utils.to_categorical(y, num_classes=num_techniques)
            
            # Build sequential model
            model = keras.Sequential([
                keras.layers.Embedding(input_dim=num_techniques, output_dim=32, input_length=seq_length),
                keras.layers.LSTM(64, return_sequences=True),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(64),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(num_techniques, activation='softmax')
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            model.fit(
                X, y_onehot,
                epochs=50,
                batch_size=8,
                validation_split=0.2,
                verbose=1
            )
            
            # Save model
            model.save(os.path.join(self.models_dir, 'attack_pattern_model.h5'))
            
            self.attack_pattern_model = model
            logger.info("Successfully trained attack pattern model")
            
            return True
        
        except Exception as e:
            logger.error(f"Error training attack pattern model: {e}")
            return False
    
    def train_target_prediction_model(self, data):
        """Train model to predict likely targets based on threat intelligence"""
        # Extract indicators
        indicators = data.get('indicators', [])
        
        if not indicators or len(indicators) < 50:
            logger.warning("Insufficient indicator data for target prediction model")
            return False
        
        try:
            # Extract features from indicators
            features = []
            targets = []
            
            # Define target sectors
            target_sectors = [
                'government', 'financial', 'healthcare', 'education', 'energy',
                'manufacturing', 'technology', 'telecommunications', 'transportation',
                'retail', 'media', 'military', 'critical infrastructure'
            ]
            
            # Process indicators
            for indicator in indicators:
                # Try to determine target sector from context
                pulse_name = indicator.get('pulse_name', '').lower()
                pulse_tags = [tag.lower() for tag in indicator.get('pulse_tags', [])]
                indicator_type = indicator.get('type', '').lower()
                threat_type = indicator.get('threat', '').lower() if 'threat' in indicator else ''
                
                # Create feature vector
                feature = [0] * len(target_sectors)
                
                for i, sector in enumerate(target_sectors):
                    # Check if sector is mentioned
                    if sector in pulse_name or sector in ' '.join(pulse_tags) or sector in threat_type:
                        feature[i] = 1
                
                # Only use if we found at least one target sector
                if sum(feature) > 0:
                    features.append(feature)
                    
                    # Add target vector (same as feature for training)
                    targets.append(feature)
            
            if len(features) < 10:
                logger.warning("Insufficient processed indicator data for target prediction")
                return False
            
            # Convert to numpy arrays
            X = np.array(features)
            y = np.array(targets)
            
            # Train model
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Save model
            with open(os.path.join(self.models_dir, 'target_prediction_model.pkl'), 'wb') as f:
                pickle.dump({
                    'model': model,
                    'target_sectors': target_sectors
                }, f)
            
            self.target_prediction_model = {
                'model': model,
                'target_sectors': target_sectors
            }
            logger.info("Successfully trained target prediction model")
            
            return True
        
        except Exception as e:
            logger.error(f"Error training target prediction model: {e}")
            return False
    
    def train_all_models(self):
        """Train all predictive models"""
        # Collect all available data
        data = self.collect_all_data()
        
        # Train models
        time_series_success = self.train_time_series_model(data)
        attack_pattern_success = self.train_attack_pattern_model(data)
        target_prediction_success = self.train_target_prediction_model(data)
        
        logger.info(f"Model training results: Time series: {time_series_success}, Attack patterns: {attack_pattern_success}, Target prediction: {target_prediction_success}")
        
        return time_series_success or attack_pattern_success or target_prediction_success
    
    def predict_future_attacks(self, forecast_days=30):
        """Generate prediction of future cyber attacks"""
        # Check if models are available
        if not self.time_series_model and not self.attack_pattern_model and not self.target_prediction_model:
            logger.warning("No trained models available for prediction")
            return None
        
        # Create forecast container
        forecast = {
            "generated_at": datetime.now().isoformat(),
            "forecast_period": {
                "start_date": datetime.now().isoformat(),
                "end_date": (datetime.now() + timedelta(days=forecast_days)).isoformat(),
                "days": forecast_days
            },
            "predictions": []
        }
        
        # Time series forecast
        if self.time_series_model:
            try:
                # Generate future dates
                future = self.time_series_model.make_future_dataframe(periods=forecast_days)
                
                # Make prediction
                forecast_result = self.time_series_model.predict(future)
                
                # Extract forecast for future dates
                future_forecast = forecast_result[forecast_result['ds'] >= datetime.now()]
                
                # Get threshold for significant days
                baseline = np.mean(forecast_result['yhat'][:len(forecast_result)-forecast_days])
                threshold = baseline * 1.5  # 50% increase from baseline
                
                # Find days with significantly elevated activity
                high_activity_days = future_forecast[future_forecast['yhat'] > threshold]
                
                # Add to predictions
                for _, day in high_activity_days.iterrows():
                    # Calculate confidence based on the prediction interval
                    confidence = max(0.5, min(0.95, 1 - ((day['yhat_upper'] - day['yhat_lower']) / day['yhat'])))
                    
                    forecast["predictions"].append({
                        "type": "activity_spike",
                        "date": day['ds'].strftime('%Y-%m-%d'),
                        "confidence": float(confidence),
                        "predicted_activity_level": float(day['yhat']),
                        "baseline_activity": float(baseline),
                        "percent_increase": float(((day['yhat'] - baseline) / baseline) * 100),
                        "analysis": "Based on historical patterns, elevated threat activity is predicted on this date"
                    })
                
                logger.info(f"Generated time series forecast with {len(high_activity_days)} elevated activity days")
            
            except Exception as e:
                logger.error(f"Error generating time series forecast: {e}")
        
        # Target sector predictions
        if self.target_prediction_model:
            try:
                model = self.target_prediction_model['model']
                target_sectors = self.target_prediction_model['target_sectors']
                
                # Make prediction based on recent threat data
                data = self.collect_all_data()
                indicators = data.get('indicators', [])
                
                if indicators:
                    # Get recent indicators (last 7 days)
                    recent_indicators = []
                    for indicator in indicators:
                        created_date = None
                        if 'created' in indicator:
                            try:
                                if 'T' in indicator['created']:
                                    created_date = datetime.fromisoformat(indicator['created'].replace('Z', '+00:00'))
                                else:
                                    created_date = datetime.strptime(indicator['created'], '%Y-%m-%d')
                            except:
                                pass
                        
                        if created_date and (datetime.now() - created_date).days <= 7:
                            recent_indicators.append(indicator)
                    
                    if recent_indicators:
                        # Count indicator types
                        indicator_types = {}
                        for indicator in recent_indicators:
                            ind_type = indicator.get('type', '')
                            indicator_types[ind_type] = indicator_types.get(ind_type, 0) + 1
                        
                        # Get top indicator type
                        top_type = max(indicator_types.items(), key=lambda x: x[1])[0] if indicator_types else ''
                        
                        # Create feature vector based on recent activity
                        feature = [0] * len(target_sectors)
                        
                        # Use recent indicators to set feature values
                        for indicator in recent_indicators:
                            pulse_name = indicator.get('pulse_name', '').lower()
                            pulse_tags = [tag.lower() for tag in indicator.get('pulse_tags', [])]
                            threat_type = indicator.get('threat', '').lower() if 'threat' in indicator else ''
                            
                            for i, sector in enumerate(target_sectors):
                                if sector in pulse_name or sector in ' '.join(pulse_tags) or sector in threat_type:
                                    feature[i] += 1
                        
                        # Normalize feature
                        total = sum(feature)
                        if total > 0:
                            feature = [x/total for x in feature]
                        
                        # Make prediction
                        predicted_targets = model.predict_proba([feature])[0]
                        
                        # Get top predicted targets (with at least 20% probability)
                        top_targets = []
                        for i, prob in enumerate(predicted_targets):
                            if prob >= 0.2:
                                top_targets.append((target_sectors[i], prob))
                        
                        # Add to predictions
                        if top_targets:
                            forecast["predictions"].append({
                                "type": "targeted_sectors",
                                "timeframe": "next_30_days",
                                "targets": [{"sector": sector.capitalize(), "probability": float(prob)} for sector, prob in top_targets],
                                "analysis": f"Based on recent threat activity, these sectors are likely to be targeted",
                                "dominant_indicator_type": top_type,
                                "recent_indicators_analyzed": len(recent_indicators)
                            })
                            
                            logger.info(f"Generated target sector predictions with {len(top_targets)} high-probability targets")
                    else:
                        logger.warning("No recent indicators found for target prediction")
                else:
                    logger.warning("No indicators available for target prediction")
            
            except Exception as e:
                logger.error(f"Error generating target sector predictions: {e}")
        
        # Add attack pattern predictions
        if self.attack_pattern_model:
            try:
                # Load technique mapping
                technique_mapping_path = os.path.join(self.models_dir, 'technique_mapping.json')
                if os.path.exists(technique_mapping_path):
                    with open(technique_mapping_path, 'r') as f:
                        technique_mapping = json.load(f)
                    
                    # Create reverse mapping
                    reverse_mapping = {v: k for k, v in technique_mapping.items()}
                    
                    # Get MITRE data
                    mitre_data = self.collect_mitre_data()
                    techniques = mitre_data.get('techniques', [])
                    
                    # Get most frequently used techniques (for starting sequences)
                    technique_usage = {}
                    for tech_id in technique_mapping.keys():
                        technique_usage[tech_id] = 0
                    
                    relationships = mitre_data.get('relationships', [])
                    for rel in relationships:
                        if rel.get('relationship_type') == 'uses' and rel.get('target_ref', '').startswith('attack-pattern--'):
                            # Find technique ID
                            tech_id = None
                            target_ref = rel.get('target_ref', '')
                            for technique in techniques:
                                if technique.get('id') == target_ref:
                                    for ref in technique.get('external_references', []):
                                        if ref.get('source_name') == 'mitre-attack':
                                            tech_id = ref.get('external_id')
                                            break
                                    break
                            
                            if tech_id and tech_id in technique_usage:
                                technique_usage[tech_id] += 1
                    
                    # Get top 3 techniques
                    top_techniques = sorted(technique_usage.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    # Generate attack chains starting from top techniques
                    attack_chains = []
                    
                    for tech_id, _ in top_techniques:
                        if tech_id in technique_mapping:
                            # Create starting sequence
                            start_seq = [technique_mapping[tech_id]]
                            
                            # Generate chain
                            chain = [tech_id]
                            
                            # Generate next 2 techniques
                            current_seq = start_seq.copy()
                            for _ in range(2):
                                # Pad sequence if needed
                                while len(current_seq) < 3:
                                    current_seq.insert(0, 0)  # Padding
                                
                                # Prepare input
                                input_seq = np.array([current_seq[-3:]])
                                
                                # Make prediction
                                prediction = self.attack_pattern_model.predict(input_seq)[0]
                                
                                # Get top 3 predictions
                                top_indices = prediction.argsort()[-3:][::-1]
                                
                                # Choose highest probability technique
                                next_tech_idx = top_indices[0]
                                if next_tech_idx in reverse_mapping:
                                    next_tech_id = reverse_mapping[next_tech_idx]
                                    chain.append(next_tech_id)
                                    current_seq.append(next_tech_idx)
                                else:
                                    break
                            
                            if len(chain) > 1:
                                attack_chains.append(chain)
                    
                    # Convert chains to readable format
                    readable_chains = []
                    for chain in attack_chains:
                        readable_chain = []
                        for tech_id in chain:
                            # Find technique details
                            for technique in techniques:
                                for ref in technique.get('external_references', []):
                                    if ref.get('source_name') == 'mitre-attack' and ref.get('external_id') == tech_id:
                                        readable_chain.append({
                                            "id": tech_id,
                                            "name": technique.get('name', ''),
                                            "description": technique.get('description', '')[:200] + '...' if len(technique.get('description', '')) > 200 else technique.get('description', '')
                                        })
                                        break
                        
                        if len(readable_chain) > 1:
                            readable_chains.append(readable_chain)
                    
                    if readable_chains:
                        forecast["predictions"].append({
                            "type": "attack_chains",
                            "timeframe": "next_30_days",
                            "chains": readable_chains,
                            "analysis": "Based on historical APT behavior, these attack chains are likely to be observed in upcoming campaigns"
                        })
                        
                        logger.info(f"Generated {len(readable_chains)} attack chains")
                
                else:
                    logger.warning("Technique mapping not found for attack pattern prediction")
            
            except Exception as e:
                logger.error(f"Error generating attack pattern predictions: {e}")
        
        # Check if we have any predictions
        if not forecast["predictions"]:
            logger.warning("No predictions generated")
            return None
        
        # Save forecast
        forecast_path = os.path.join(self.forecasts_dir, f"forecast_{datetime.now().strftime('%Y%m%d')}.json")
        with open(forecast_path, 'w') as f:
            json.dump(forecast, f, indent=2)
        
        logger.info(f"Generated forecast with {len(forecast['predictions'])} predictions")
        
        return forecast

# Main execution
if __name__ == "__main__":
    predictor = FutureAttackPredictor()
    # Train models if needed
    if not predictor.time_series_model or not predictor.attack_pattern_model or not predictor.target_prediction_model:
        logger.info("Training models...")
        predictor.train_all_models()
    # Generate forecast
    forecast = predictor.predict_future_attacks()
    print(f"Generated forecast saved to {predictor.forecasts_dir}")
EOT

chmod +x models/predictive/future_attack_predictor.py

# Step 4: Create main enhancement script
echo "Creating main enhancement script..."
cat > scripts/predictive/enhance_system.py << 'EOT'
#!/usr/bin/env python3
import os
import json
import subprocess
import logging
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhance_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EnhanceSystem')

def check_dependencies():
    """Check if required dependencies are installed"""
    dependencies = [
        "pandas", "scikit-learn", "numpy", "tensorflow", "keras", 
        "prophet", "nltk", "requests-html", "feedparser"
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        logger.warning(f"Missing dependencies: {', '.join(missing)}")
        return False
    
    logger.info("All dependencies are installed")
    return True

def collect_real_data():
    """Collect real threat intelligence data"""
    logger.info("Starting real data collection...")
    
    try:
        # Run data collector
        result = subprocess.run(
            ["python3", "scripts/predictive/real_data_collector.py"],
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info(f"Data collection complete: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Data collection failed: {e.stderr}")
        return False

def train_prediction_models():
    """Train predictive models using collected data"""
    logger.info("Starting model training...")
    
    try:
        # Run model trainer
        result = subprocess.run(
            ["python3", "models/predictive/future_attack_predictor.py"],
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info(f"Model training complete: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Model training failed: {e.stderr}")
        return False

def create_prediction_endpoint():
    """Create prediction endpoint in API server"""
    logger.info("Adding prediction endpoint to API server...")
    
    # Check if api_server.py exists
    if not os.path.exists('api_server.py'):
        logger.error("api_server.py not found")
        return False
    
    # Read api_server.py
    with open('api_server.py', 'r') as f:
        content = f.read()
    
    # Check if prediction endpoint already exists
    if "/predict_attacks" in content:
        logger.info("Prediction endpoint already exists")
        return True
    
    # Add import for future attack prediction
    import_line = "from models.predictive.future_attack_predictor import FutureAttackPredictor"
    
    # Find import section
    import_section_end = content.find("# Configuration")
    if import_section_end == -1:
        import_section_end = content.find("# Initialize flag")
    
    if import_section_end == -1:
        logger.error("Could not find import section")
        return False
    
    # Insert import
    content = content[:import_section_end] + import_line + "\n" + content[import_section_end:]
    
    # Initialize predictor
    init_line = "\n# Initialize attack predictor\ntry:\n    attack_predictor = FutureAttackPredictor()\n    print(\"Initialized future attack prediction engine\")\nexcept Exception as e:\n    print(f\"Error initializing attack predictor: {e}\")\n    attack_predictor = None\n"
    
    # Find initialization section
    init_section_end = content.find("# Tracking stats")
    if init_section_end == -1:
        logger.error("Could not find initialization section")
        return False
    
    # Insert initialization
    content = content[:init_section_end] + init_line + content[init_section_end:]
    
    # Add prediction endpoint
    endpoint_code = """
        # Future attack prediction endpoint
        elif path == "/predict_attacks":
            self._set_headers()
            
            # Check if attack predictor is available
            if attack_predictor is None:
                self.send_response(503)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Attack prediction engine not available",
                    "timestamp": datetime.now().isoformat()
                }).encode())
                stats["errors"] += 1
                return
            
            # Parse query parameters
            parsed_query = urllib.parse.parse_qs(parsed_path.query)
            forecast_days = int(parsed_query.get("days", ["30"])[0])
            
            try:
                # Generate attack forecast
                forecast = attack_predictor.predict_future_attacks(forecast_days=forecast_days)
                
                if forecast:
                    self.wfile.write(json.dumps(forecast).encode())
                else:
                    self.wfile.write(json.dumps({
                        "error": "Could not generate forecast",
                        "timestamp": datetime.now().isoformat()
                    }).encode())
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": f"Error generating forecast: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }).encode())
                stats["errors"] += 1
"""
    
    # Find endpoint section
    endpoint_section = content.find("# Status endpoint")
    if endpoint_section == -1:
        logger.error("Could not find endpoint section")
        return False
    
    # Move to the end of the status endpoint code
    endpoint_insertion_point = content.find("return", endpoint_section)
    endpoint_insertion_point = content.find("\n", endpoint_insertion_point)
    
    if endpoint_insertion_point == -1:
        logger.error("Could not find insertion point for endpoint")
        return False
    
    # Insert endpoint code
    content = content[:endpoint_insertion_point + 1] + endpoint_code + content[endpoint_insertion_point + 1:]
    
    # Add endpoint to homepage API info
    home_endpoints = content.find("\"endpoints\": [")
    if home_endpoints == -1:
        logger.error("Could not find home endpoints section")
        return False
    
    # Find the end of the endpoints list
    endpoints_end = content.find("]", home_endpoints)
    if endpoints_end == -1:
        logger.error("Could not find endpoints list end")
        return False
    
    # Insert new endpoint
    endpoint_entry = '                    {"path": "/predict_attacks", "method": "GET", "description": "Predict future cyber attacks"},'
    content = content[:endpoints_end] + endpoint_entry + content[endpoints_end:]
    
    # Write updated content
    with open('api_server.py', 'w') as f:
        f.write(content)
    
    logger.info("Successfully added prediction endpoint to API server")
    return True

def main():
    logger.info("Starting Echelon enhancement process")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing dependencies, please install required packages")
        sys.exit(1)
    
    # Collect real data
    if not collect_real_data():
        logger.error("Failed to collect real data")
        sys.exit(1)
    
    # Train prediction models
    if not train_prediction_models():
        logger.error("Failed to train prediction models")
        sys.exit(1)
    
    # Create prediction endpoint
    if not create_prediction_endpoint():
        logger.error("Failed to create prediction endpoint")
        sys.exit(1)
    
    logger.info("Enhancement process completed successfully!")
    logger.info("Echelon has been enhanced with future attack prediction capabilities using real data only")
    logger.info("To access predictions, use the /predict_attacks endpoint in the API server")

if __name__ == "__main__":
    main()
EOT

chmod +x scripts/predictive/enhance_system.py

# Create the main enhancement shell script
cat > enhance_predictive_capabilities.sh << 'EOT'
#!/bin/bash
# enhance_predictive_capabilities.sh - Transform Echelon to predict future cyber attacks using real data only

echo "========================================="
echo "ECHELON: ENHANCING PREDICTIVE CAPABILITIES"
echo "========================================="
echo "Installing advanced predictive capabilities for future attack forecasting"
echo "USING ONLY REAL DATA - NO SIMULATIONS"

# Install required Python dependencies
echo "Installing advanced dependencies..."
pip install pandas scikit-learn numpy tensorflow keras prophet nltk requests-html feedparser tqdm

# Create necessary directories
mkdir -p models/predictive
mkdir -p data/trends
mkdir -p data/raw/threat_feeds
mkdir -p data/processed/attack_forecasts
mkdir -p scripts/predictive

# Enhance system by collecting data and training models
echo "Enhancing system with predictive capabilities..."
python3 scripts/predictive/enhance_system.py

# Check if enhancement was successful
if [ $? -eq 0 ]; then
    echo "========================================="
    echo "ENHANCEMENT COMPLETE!"
    echo "========================================="
    echo ""
    echo "Echelon has been enhanced with the ability to predict future"
    echo "cyber attacks based on real-world threat intelligence data."
    echo ""
    echo "New capabilities:"
    echo "- Time series forecasting of future attack activity"
    echo "- Prediction of likely attack chains and techniques"
    echo "- Identification of sectors likely to be targeted"
    echo ""
    echo "To access predictions, use the new endpoint:"
    echo "  http://localhost:8080/predict_attacks"
    echo ""
    echo "To generate fresh predictions, run:"
    echo "  python3 models/predictive/future_attack_predictor.py"
    echo ""
    echo "All predictions are based on REAL threat intelligence data"
    echo "collected from authoritative sources - NO SIMULATIONS."
else
    echo "========================================="
    echo "ENHANCEMENT FAILED"
    echo "========================================="
    echo ""
    echo "Please check the logs for more information."
    echo "You may need to configure API keys in config/api_keys.json"
    echo "to access real threat intelligence sources."
fi
EOT

chmod +x enhance_predictive_capabilities.sh

echo "========================================="
echo "ENHANCED CAPABILITIES SCRIPT CREATED"
echo "========================================="
echo ""
echo "The enhance_predictive_capabilities.sh script has been created."
echo "This script will add future attack prediction capabilities to Echelon"
echo "using only real threat intelligence data from authoritative sources."
echo ""
echo "To run the enhancement, execute:"
echo "  ./enhance_predictive_capabilities.sh"
echo ""
echo "IMPORTANT: This enhancement requires API keys for threat intelligence"
echo "sources such as AlienVault OTX. Make sure these are configured in"
echo "config/api_keys.json before running the enhancement."
echo ""
echo "NO SIMULATIONS OR SYNTHETIC DATA WILL BE USED - ONLY REAL DATA."