import sqlite3
import logging
import os
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
from config import DB_PATH

def dict_factory(cursor, row):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = dict_factory
    try:
        yield conn
    finally:
        conn.close()

def execute_query(query: str, params: tuple = None, fetch: bool = False):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch:
                return cursor.fetchall()
            else:
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            conn.rollback()
            raise

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS threats (
            id TEXT PRIMARY KEY,
            source TEXT,
            title TEXT,
            description TEXT,
            published_date TEXT,
            tags TEXT,
            indicators TEXT,
            attack_vector TEXT,
            regions_affected TEXT,
            sectors_affected TEXT,
            severity TEXT,
            confidence INTEGER,
            raw_data TEXT,
            processed BOOLEAN DEFAULT 0,
            created_at TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            apt_group TEXT,
            attack_type TEXT,
            threat_category TEXT,
            region TEXT,
            industry TEXT,
            severity TEXT,
            likelihood TEXT,
            confidence INTEGER,
            timestamp TEXT,
            description TEXT,
            indicators TEXT,
            affecting TEXT,
            evidence TEXT,
            created_at TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_data (
            id TEXT PRIMARY KEY,
            name TEXT,
            version TEXT,
            accuracy REAL,
            precision_avg REAL,
            recall_avg REAL,
            f1_avg REAL,
            features TEXT,
            hyperparameters TEXT,
            last_trained TEXT,
            training_duration INTEGER,
            data_points_count INTEGER,
            meta_data TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS taxonomy (
            id TEXT PRIMARY KEY,
            type TEXT,
            value TEXT,
            aliases TEXT,
            description TEXT,
            first_seen TEXT,
            last_updated TEXT,
            source TEXT,
            meta_data TEXT
        )
        ''')
        
        conn.commit()
