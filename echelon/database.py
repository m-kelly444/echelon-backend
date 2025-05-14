import sqlite3
import os
from contextlib import contextmanager

DB_PATH = os.getenv('DB_PATH', 'data/echelon.db')

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

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
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
        conn.commit()
