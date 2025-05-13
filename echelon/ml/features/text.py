import re
import numpy as np
from typing import Dict, List, Any
from sklearn.feature_extraction.text import TfidfVectorizer

CVE_PATTERN = re.compile(r'CVE-\d{4}-\d{4,7}')
APT_GROUP_PATTERN = re.compile(r'\b(?:APT|UNC|TA)\d+\b|\b(?:Lazarus|Cozy\s*Bear|Fancy\s*Bear|Sandworm|Kimsuky|Wizard\s*Spider)\b', re.IGNORECASE)
IP_PATTERN = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
HASH_PATTERN = re.compile(r'\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b')

class TextFeatureExtractor:
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.fitted = False
    
    def fit(self, texts):
        self.vectorizer.fit(texts)
        self.fitted = True
        return self
    
    def transform(self, text):
        if not self.fitted:
            raise ValueError("TextFeatureExtractor must be fitted before transform")
        return self.vectorizer.transform([text]).toarray()[0]
    
    def extract_entities(self, text):
        entities = {
            'cves': list(set(CVE_PATTERN.findall(text))),
            'apt_groups': list(set(APT_GROUP_PATTERN.findall(text))),
            'ips': list(set(IP_PATTERN.findall(text))),
            'hashes': list(set(HASH_PATTERN.findall(text)))
        }
        return entities
