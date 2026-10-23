import sqlite3
import os
import json
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "pakcoy_smart_farming.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Plants table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS plants (
        plant_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        batch TEXT NOT NULL,
        variety TEXT DEFAULT 'Pakcoy Green Fortune',
        planting_date TEXT,
        current_day INTEGER DEFAULT 1,
        current_dap INTEGER DEFAULT 10,
        status TEXT DEFAULT 'Normal',
        notes TEXT
    )
    """)
    
    # 2. Observations table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plant_id TEXT NOT NULL,
        day INTEGER NOT NULL,
        dap INTEGER NOT NULL,
        time TEXT NOT NULL,
        soil_moisture REAL NOT NULL,
        temperature REAL NOT NULL,
        soil_condition TEXT NOT NULL,
        maturity_level REAL NOT NULL,
        criteria TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(plant_id) REFERENCES plants(plant_id)
    )
    """)
    
    # 3. CNN Results table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cnn_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plant_id TEXT,
        image_path TEXT NOT NULL,
        predicted_class TEXT NOT NULL,
        confidence REAL NOT NULL,
        probabilities_json TEXT,
        gradcam_path TEXT,
        analyzed_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # 4. LSTM Predictions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS lstm_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plant_id TEXT NOT NULL,
        base_day INTEGER NOT NULL,
        forecast_json TEXT NOT NULL,
        metrics_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # 5. Anomalies & Alerts table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plant_id TEXT NOT NULL,
        day INTEGER,
        alert_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        status TEXT DEFAULT 'Active',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # 6. Chat History
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        plant_id TEXT,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        tool_calls_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    conn.close()
    print("Database initialized successfully at:", DB_PATH)

if __name__ == "__main__":
    init_db()
