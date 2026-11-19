import pandas as pd
import json
from database.db import get_connection

def get_all_plants():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM plants ORDER BY plant_id", conn)
    conn.close()
    return df

def get_plant_details(plant_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM plants WHERE plant_id = ?", (plant_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_growth_history(plant_id: str, limit: int = 5000):
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM observations WHERE plant_id = ? ORDER BY day ASC, time ASC LIMIT ?",
        conn,
        params=(plant_id, limit)
    )
    conn.close()
    return df

def get_latest_observation(plant_id: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM observations WHERE plant_id = ? ORDER BY day DESC, time DESC LIMIT 1",
        (plant_id,)
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_alerts(plant_id: str = None, status: str = "Active"):
    conn = get_connection()
    query = "SELECT * FROM alerts WHERE 1=1"
    params = []
    if plant_id:
        query += " AND plant_id = ?"
        params.append(plant_id)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY id DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def get_plant_comparison(plant_id1: str, plant_id2: str):
    p1 = get_plant_details(plant_id1)
    p2 = get_plant_details(plant_id2)
    obs1 = get_latest_observation(plant_id1)
    obs2 = get_latest_observation(plant_id2)
    
    return {
        plant_id1: {"info": p1, "latest": obs1},
        plant_id2: {"info": p2, "latest": obs2}
    }

def get_all_observations_df():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM observations ORDER BY plant_id, day, time", conn)
    conn.close()
    return df

def add_observation_record(plant_id, day, dap, time, soil_moisture, temperature, soil_condition, maturity_level, criteria):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO observations (plant_id, day, dap, time, soil_moisture, temperature, soil_condition, maturity_level, criteria)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (plant_id, day, dap, time, soil_moisture, temperature, soil_condition, maturity_level, criteria))
    
    # Update plant current_day and current_dap
    cursor.execute("""
    UPDATE plants 
    SET current_day = ?, current_dap = ?
    WHERE plant_id = ?
    """, (day, dap, plant_id))
    
    conn.commit()
    conn.close()
    return True

def get_activity_logs(filter_type: str = "Semua"):
    conn = get_connection()
    if filter_type == "Semua" or not filter_type:
        query = "SELECT * FROM activity_logs ORDER BY id DESC"
        df = pd.read_sql_query(query, conn)
    else:
        query = "SELECT * FROM activity_logs WHERE jenis = ? ORDER BY id DESC"
        df = pd.read_sql_query(query, conn, params=(filter_type,))
    conn.close()
    return df

def add_activity_log(jenis: str, proses: str, status: str = "Berhasil", detail: dict = None):
    from datetime import datetime
    waktu_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO activity_logs (waktu, jenis, proses, status, detail_json)
    VALUES (?, ?, ?, ?, ?)
    """, (waktu_str, jenis, proses, status, json.dumps(detail or {})))
    conn.commit()
    conn.close()
    return True
