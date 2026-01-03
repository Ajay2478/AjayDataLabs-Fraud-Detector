import sqlite3
from datetime import datetime

DB_NAME = "fraud_detection.db"

def init_db():
    """Creates the table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            amount REAL,
            anomaly_score REAL,
            prediction TEXT,
            is_fraud INTEGER
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Database initialized.")

def save_transaction(amount, score, prediction, is_fraud):
    """Saves a single processed transaction."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO transactions (timestamp, amount, anomaly_score, prediction, is_fraud)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, amount, score, prediction, int(is_fraud)))
    
    conn.commit()
    conn.close()