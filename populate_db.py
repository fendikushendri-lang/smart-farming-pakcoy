import os
import pandas as pd
from database.db import get_connection, init_db

def populate(force=False):
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if already populated
    cursor.execute("SELECT COUNT(*) FROM observations")
    count = cursor.fetchone()[0]
    if count > 0 and not force:
        print("Database observations already populated.")
        conn.close()
        return

    if force:
        cursor.execute("DELETE FROM observations")
        cursor.execute("DELETE FROM plants")
        cursor.execute("DELETE FROM alerts")
        conn.commit()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "data", "processed", "timeseries_clean.csv")
    df = pd.read_csv(csv_path)
    
    # Define Plant / Batch configurations
    # All strictly conform to the maximum 40-day harvest timeline (Maksimal 40 Hari Panen)
    # anchored directly to the real timeseries_clean.csv rows (Day 1 = 10 HST, Day 31 = 40 HST)
    plants_info = [
        {
            "plant_id": "PCK-001",
            "name": "Pakcoy Hidroponik NFT A-1",
            "batch": "Batch 2026-NFT-01",
            "variety": "Pakcoy Green Fortune",
            "start_day": 1,
            "end_day": 28,
            "notes": "Siklus tanam hidroponik NFT fase Siap Panen Prima (37 HST). Sisa 3 hari sebelum mencapai batas maksimal panen 40 HST."
        },
        {
            "plant_id": "PCK-002",
            "name": "Pakcoy Polybag B-2 (Vegetatif)",
            "batch": "Batch 2026-SUB-02",
            "variety": "Pakcoy Nauli F1",
            "start_day": 1,
            "end_day": 16,
            "notes": "Siklus tanam substrat polybag fase Vegetatif aktif (25 HST). Pertumbuhan daun pesat, sisa 15 hari menuju batas panen 40 HST."
        },
        {
            "plant_id": "PCK-003",
            "name": "Pakcoy DWC C-1 (Pre-Harvest)",
            "batch": "Batch 2026-DWC-03",
            "variety": "Pakcoy Green Fortune",
            "start_day": 1,
            "end_day": 23,
            "notes": "Sistem rakit apung DWC fase Pre-Harvest (32 HST). Kanopi daun memadat, sisa 8 hari menuju batas panen 40 HST."
        },
        {
            "plant_id": "PCK-004",
            "name": "Pakcoy Vertikal Tower D-2 (Batas Panen)",
            "batch": "Batch 2026-AERO-04",
            "variety": "Pakcoy Nauli F1",
            "start_day": 1,
            "end_day": 31,
            "notes": "Sistem aeroponik vertikal tower telah mencapai batas akhir panen 40 HST (Day 31 = DAP 40). Wajib dipanen hari ini!"
        },
        {
            "plant_id": "PCK-005",
            "name": "Pakcoy Semai Polybag E-1",
            "batch": "Batch 2026-SEM-05",
            "variety": "Pakcoy Green Fortune",
            "start_day": 1,
            "end_day": 5,
            "notes": "Fase semai dan persemaian awal (14 HST). Daun kotiledon tumbuh seragam, sisa 26 hari menuju batas panen 40 HST."
        },
        {
            "plant_id": "PCK-FULL40",
            "name": "Pakcoy Siklus Lengkap 40 Hari",
            "batch": "Batch 2026-FULL-40D",
            "variety": "Pakcoy Green Fortune",
            "start_day": 1,
            "end_day": 40,
            "notes": "Siklus pengamatan kontinu 40 hari penuh (120 observasi, Day 1-40, DAP 10-49 HST) mencakup seluruh tahapan budidaya."
        }
    ]
    
    # Insert Plants
    for p in plants_info:
        sub_df = df[(df['day'] >= p['start_day']) & (df['day'] <= p['end_day'])]
        latest_row = sub_df.iloc[-1]
        
        # Determine status (Mark as Perlu Perhatian if non-optimal soil or reached harvest deadline 40 HST)
        status = "Optimal"
        if latest_row['soil_condition'] != 'Optimal' or latest_row['dap'] >= 40:
            status = "Perlu Perhatian"
        
        cursor.execute("""
        INSERT OR REPLACE INTO plants (plant_id, name, batch, variety, planting_date, current_day, current_dap, status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            p['plant_id'],
            p['name'],
            p['batch'],
            p['variety'],
            f"2026-05-{p['start_day']:02d}" if p['start_day'] <= 30 else f"2026-06-01",
            int(latest_row['day']),
            int(latest_row['dap']),
            status,
            p['notes']
        ))
        
        # Insert observations for each plant
        obs_records = []
        for _, row in sub_df.iterrows():
            obs_records.append((
                p['plant_id'],
                int(row['day']),
                int(row['dap']),
                str(row['time']),
                float(row['soil_moisture']),
                float(row['temperature']),
                str(row['soil_condition']),
                float(row['maturity_level']),
                str(row['growth_stage'])
            ))
            
        cursor.executemany("""
        INSERT INTO observations (plant_id, day, dap, time, soil_moisture, temperature, soil_condition, maturity_level, criteria)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, obs_records)
    
    # Insert initial anomalies/alerts based on actual data
    anomalies = [
        ("PCK-001", 2, "Soil Condition", "Medium", "Kelembaban Tinggi (Wet)", "Kelembaban tanah tercatat 76% (Wet condition) pada jam 08:00 WIB, melebihi batas optimal pakcoy."),
        ("PCK-001", 11, "Soil Condition", "High", "Kekeringan Media Tanam (Dry)", "Kelembaban tanah tercatat 63% (Dry condition) pada jam 08:00 WIB, membutuhkan perhatian penyiraman."),
        ("PCK-004", 31, "Masa Panen", "High", "Wajib Panen Hari Ini (40 HST)", "Tanaman PCK-004 telah mencapai batas usia panen maksimal pakcoy yaitu 40 HST. Segera lakukan pemanenan untuk menjaga kualitas kerenyahan pelepah daun.")
    ]
    
    for a in anomalies:
        cursor.execute("""
        INSERT INTO alerts (plant_id, day, alert_type, severity, title, description, status)
        VALUES (?, ?, ?, ?, ?, ?, 'Active')
        """, a)
    
    conn.commit()
    conn.close()
    print("Database successfully populated with plant profiles and observations.")

if __name__ == "__main__":
    populate(force=True)
