"""
SQLite Database Storage for Industrial Inspection History and Audit Logging
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from src.config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "backend" / "inspections.db"

def get_db_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inspections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        filename TEXT NOT NULL,
        primary_prediction TEXT NOT NULL,
        primary_confidence REAL NOT NULL,
        consensus_count INTEGER NOT NULL,
        total_models INTEGER NOT NULL,
        consensus_status TEXT NOT NULL,
        confidence_tier TEXT NOT NULL,
        recommendation TEXT NOT NULL,
        custom_pred TEXT,
        custom_conf REAL,
        frozen_pred TEXT,
        frozen_conf REAL,
        finetuned_pred TEXT,
        finetuned_conf REAL
    )
    """)
    conn.commit()
    conn.close()

def log_inspection(
    filename,
    primary_prediction,
    primary_confidence,
    consensus_count,
    total_models,
    consensus_status,
    confidence_tier,
    recommendation,
    models_dict,
):
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c_p = models_dict.get("custom_cnn", {}).get("predicted_class")
    c_c = models_dict.get("custom_cnn", {}).get("confidence")
    fz_p = models_dict.get("efficientnet_frozen", {}).get("predicted_class")
    fz_c = models_dict.get("efficientnet_frozen", {}).get("confidence")
    ft_p = models_dict.get("efficientnet_finetuned", {}).get("predicted_class")
    ft_c = models_dict.get("efficientnet_finetuned", {}).get("confidence")

    cursor.execute("""
    INSERT INTO inspections (
        timestamp, filename, primary_prediction, primary_confidence,
        consensus_count, total_models, consensus_status, confidence_tier, recommendation,
        custom_pred, custom_conf, frozen_pred, frozen_conf, finetuned_pred, finetuned_conf
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str, filename, primary_prediction, primary_confidence,
        consensus_count, total_models, consensus_status, confidence_tier, recommendation,
        c_p, c_c, fz_p, fz_c, ft_p, ft_c
    ))
    inspection_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inspection_id, now_str

def get_recent_inspections(limit=30):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM inspections ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Initialize table on import
init_db()
