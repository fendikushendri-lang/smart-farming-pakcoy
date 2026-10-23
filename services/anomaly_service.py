import pandas as pd
import numpy as np
from database.db import get_connection

def detect_anomalies_for_plant(plant_id: str):
    """
    Menganalisis anomali pertumbuhan dan lingkungan untuk tanaman tertentu
    berdasarkan data historis dan ambang batas agronomi pakcoy.
    """
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM observations WHERE plant_id = ? ORDER BY day ASC, time ASC",
        conn,
        params=(plant_id,)
    )
    conn.close()
    
    if df.empty:
        return []
    
    anomalies = []
    
    # 1. Soil Condition check
    non_optimal = df[df['soil_condition'] != 'Optimal']
    for _, row in non_optimal.iterrows():
        severity = "High" if row['soil_condition'] == 'Dry' else "Medium"
        desc = (
            f"Kondisi tanah '{row['soil_condition']}' tercatat pada Hari ke-{row['day']} (DAP {row['dap']}) pukul {row['time']}. "
            f"Kelembaban: {row['soil_moisture']}%. Perlu intervensi pengaturan irigasi."
        )
        anomalies.append({
            "plant_id": plant_id,
            "day": int(row['day']),
            "time": row['time'],
            "type": "Soil Moisture",
            "severity": severity,
            "title": f"Anomali Kelembaban: {row['soil_condition']}",
            "description": desc,
            "value": f"{row['soil_moisture']}%"
        })
        
    # 2. Temperature Threshold check
    high_temp = df[df['temperature'] > 30.0]
    for _, row in high_temp.iterrows():
        anomalies.append({
            "plant_id": plant_id,
            "day": int(row['day']),
            "time": row['time'],
            "type": "Temperature",
            "severity": "Medium",
            "title": "Suhu Lingkungan Tinggi",
            "description": f"Suhu lingkungan mencapai {row['temperature']}°C pada Hari ke-{row['day']} pukul {row['time']} (ambang ideal pakcoy: 22-28°C).",
            "value": f"{row['temperature']}°C"
        })
        
    # 3. Growth rate check (Check for stagnation)
    daily_mat = df.groupby('day')['maturity_level'].last()
    if len(daily_mat) >= 5:
        # Check last 4 days growth
        diff_last_4 = daily_mat.iloc[-1] - daily_mat.iloc[-5] if len(daily_mat) >= 5 else 0
        latest_day = daily_mat.index[-1]
        latest_dap = df[df['day'] == latest_day]['dap'].iloc[-1]
        
        # In vegetative stage (DAP 20-30), maturity should progress
        if latest_dap <= 34 and diff_last_4 < 1.0:
            anomalies.append({
                "plant_id": plant_id,
                "day": int(latest_day),
                "time": "Daily Check",
                "type": "Growth Rate",
                "severity": "High",
                "title": "Laju Pertumbuhan Stagnan",
                "description": f"Pertumbuhan maturity level lambat (+{diff_last_4:.1f}%) dalam 4 hari terakhir pada fase kritis.",
                "value": f"+{diff_last_4:.1f}%"
            })
            
    # 4. Harvest Deadline Check (Masa Panen Maksimal 40 Hari)
    latest_row = df.iloc[-1]
    curr_dap = int(latest_row['dap'])
    curr_day = int(latest_row['day'])
    
    if curr_dap >= 40:
        anomalies.append({
            "plant_id": plant_id,
            "day": curr_day,
            "time": "Harvest Alert",
            "type": "Masa Panen",
            "severity": "High",
            "title": "Batas Maksimal Panen 40 HST Tercapai!",
            "description": f"Tanaman telah berumur {curr_dap} HST (batas maksimal panen pakcoy adalah 40 HST). Wajib segera dipanen hari ini untuk mencegah serat mengeras dan mencegah pembungaan (bolting).",
            "value": f"{curr_dap} / 40 HST"
        })
    elif curr_dap >= 35:
        anomalies.append({
            "plant_id": plant_id,
            "day": curr_day,
            "time": "Harvest Window",
            "type": "Masa Panen",
            "severity": "Medium",
            "title": f"Jendela Panen Prima (Sisa {40 - curr_dap} Hari)",
            "description": f"Tanaman berumur {curr_dap} HST telah memasuki fase siap panen optimal (35-40 HST). Sisa waktu sebelum batas akhir 40 HST: {40 - curr_dap} hari.",
            "value": f"{curr_dap} / 40 HST"
        })
            
    return anomalies

def get_anomaly_summary():
    """Mengambil ringkasan anomali untuk semua tanaman aktif."""
    conn = get_connection()
    alerts_df = pd.read_sql_query("SELECT * FROM alerts WHERE status = 'Active'", conn)
    conn.close()
    return alerts_df
