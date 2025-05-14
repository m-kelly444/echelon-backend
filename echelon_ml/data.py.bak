import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from sklearn.model_selection import train_test_split

class ThreatDataset:
    def __init__(self, data_source, test_size=0.2, random_state=42):
        self.data_source = data_source
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.features = None
        self.labels = None
        self.train_data = None
        self.test_data = None
        self.train_labels = None
        self.test_labels = None
        
        self._load_data()
        self._split_data()
    
    def _load_data(self):
        if isinstance(self.data_source, str):
            if self.data_source.endswith('.csv'):
                self.data = pd.read_csv(self.data_source)
            elif self.data_source.endswith('.json'):
                self.data = pd.read_json(self.data_source)
            else:
                raise ValueError(f"Unsupported file format: {self.data_source}")
        elif isinstance(self.data_source, pd.DataFrame):
            self.data = self.data_source
        else:
            raise ValueError(f"Unsupported data source type: {type(self.data_source)}")
        
        self._preprocess_data()
    
    def _preprocess_data(self):
        if 'label' in self.data.columns:
            self.labels = self.data['label'].values
            self.features = self.data.drop('label', axis=1).values
        else:
            raise ValueError("Data must contain a 'label' column")
    
    def _split_data(self):
        if self.features is None or self.labels is None:
            raise ValueError("Data must be loaded and preprocessed before splitting")
        
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            self.features, self.labels, test_size=self.test_size, random_state=self.random_state
        )
    
    def __len__(self):
        return len(self.data) if self.data is not None else 0

def load_data(path):
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.json'):
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")
