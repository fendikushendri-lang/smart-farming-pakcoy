import os
import pandas as pd
from database.db import get_connection, init_db

def load_dataset():
    """Memuat dataset sensorik pakcoy dari CSV atau Excel dengan fallback cerdas."""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(BASE_DIR, "data", "processed", "timeseries_clean.csv"),
        os.path.join(BASE_DIR, "timeseries_clean.csv"),
        os.path.join(BASE_DIR, "data", "timeseries_clean.csv"),
        os.path.join(BASE_DIR, "Dataset_Pakcoy_40_Hari_Clean.xlsx"),
        os.path.join(BASE_DIR, "new Dataset_Pakcoy.xlsx"),
        os.path.join(BASE_DIR, "data", "Dataset_Pakcoy_40_Hari_Clean.xlsx"),
        os.path.join(BASE_DIR, "data", "new Dataset_Pakcoy.xlsx"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                if p.endswith(".csv"):
                    df = pd.read_csv(p)
                else:
                    df_raw = pd.read_excel(p)
                    rename_map = {
                        'Day': 'day',
                        'DAP': 'dap',
                        'Time': 'time',
                        'Soil Moisture (%)': 'soil_moisture',
                        'Soil Condition': 'soil_condition',
                        'Ground Truth Maturity Level (%)': 'maturity_level',
                        'Ground Truth Criteria': 'criteria'
                    }
                    for c in df_raw.columns:
                        if 'Temperature' in c or 'Temp' in c:
                            rename_map[c] = 'temperature'
                    df = df_raw.rename(columns=rename_map)
                
                # Pastikan kolom esensial ada
                if 'growth_stage' not in df.columns:
                    df['growth_stage'] = df.get('criteria', 'Vegetatif')
                if 'timestep' not in df.columns:
                    df['timestep'] = range(1, len(df) + 1)
                return df
            except Exception as e:
                print(f"Catatan load {p}: {e}")
                continue

    # Fallback sintetis 40 hari penuh jika file fisik tidak ditemukan (anti-crash cloud)
    records = []
    for day in range(1, 41):
        dap = day + 9
        for t in ["08:00", "12:00", "16:00"]:
            moisture = 52.0 if dap < 35 else 50.0
            temp = 25.0 if t == "08:00" else (28.5 if t == "12:00" else 27.0)
            cond = "Optimal"
            mat = min(round(30.0 + (dap - 10) * 2.0, 1), 98.0)
            crit = "Initial Stage" if dap < 20 else ("Vegetative Stage" if dap < 35 else "Harvest Ready")
            records.append({
                "day": day, "dap": dap, "time": t,
                "soil_moisture": moisture, "temperature": temp,
                "soil_condition": cond, "maturity_level": mat,
                "criteria": crit, "growth_stage": crit,
                "timestep": len(records) + 1
            })
    return pd.DataFrame(records)

def populate(force=False):
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    # Cek apakah tabel observasi sudah terisi
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

    df = load_dataset()
    
    # Konfigurasi tanaman/batch 40 HST (Urut Kronologis dari PCK-001 Semai s.d. PCK-005 Panen Maksimal)
    plants_info = [
        {
            "plant_id": "PCK-001",
            "name": "Pakcoy Semai Polybag A-1 (Semai 10 HST)",
            "batch": "Batch 2026-SEM-01",
            "variety": "Pakcoy Green Fortune",
            "start_day": 1,
            "end_day": 10,
            "notes": "Fase semai dan persemaian awal (10 HST). Daun kotiledon tumbuh seragam 2-4 helai."
        },
        {
            "plant_id": "PCK-002",
            "name": "Pakcoy Polybag B-2 (Vegetatif 20 HST)",
            "batch": "Batch 2026-SUB-02",
            "variety": "Pakcoy Nauli F1",
            "start_day": 1,
            "end_day": 20,
            "notes": "Fase vegetatif aktif (20 HST) lebar daun 5.8 cm, irigasi pompa otomatis terjaga."
        },
        {
            "plant_id": "PCK-003",
            "name": "Pakcoy DWC C-1 (Pre-Harvest 30 HST)",
            "batch": "Batch 2026-DWC-03",
            "variety": "Pakcoy Green Fortune",
            "start_day": 1,
            "end_day": 30,
            "notes": "Fase mendekati panen (30 HST), tajuk roset daun rimbun 8.5 cm."
        },
        {
            "plant_id": "PCK-004",
            "name": "Pakcoy Vertikal Tower D-2 (Siap Panen 36 HST)",
            "batch": "Batch 2026-AERO-04",
            "variety": "Pakcoy Nauli F1",
            "start_day": 1,
            "end_day": 36,
            "notes": "Fase siap panen komersial optimal (36 HST), lebar daun 10.5 cm siap petik prima."
        },
        {
            "plant_id": "PCK-005",
            "name": "Pakcoy Smart IoT E-1 (Batas Panen 40 HST)",
            "batch": "Batch 2026-IoT-05",
            "variety": "Pakcoy Green Fortune",
            "start_day": 1,
            "end_day": 40,
            "notes": "Siklus batas panen maksimal pakcoy 40 HST, lebar daun 11.2 cm (Standar pasar komersial terpenuhi)."
        },
        {
            "plant_id": "PCK-FULL40",
            "name": "Pakcoy Siklus Lengkap 40 Hari",
            "batch": "Batch 2026-FULL-40D",
            "variety": "Pakcoy Green Fortune",
            "start_day": 1,
            "end_day": 40,
            "notes": "Siklus pengamatan kontinu 40 hari penuh mencakup seluruh tahapan budidaya pakcoy."
        }
    ]
    
    # Simpan daftar tanaman
    for p in plants_info:
        sub_df = df[(df['day'] >= p['start_day']) & (df['day'] <= p['end_day'])]
        if sub_df.empty:
            sub_df = df
        latest_row = sub_df.iloc[-1]
        
        status = "Optimal"
        if latest_row.get('soil_condition') != 'Optimal' or latest_row.get('dap', 0) >= 40:
            status = "Perlu Perhatian"
        
        cursor.execute("""
        INSERT OR REPLACE INTO plants (plant_id, name, batch, variety, planting_date, current_day, current_dap, status, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            p['plant_id'],
            p['name'],
            p['batch'],
            p['variety'],
            f"2026-05-{p['start_day']:02d}" if p['start_day'] <= 30 else "2026-06-01",
            int(latest_row['day']),
            int(latest_row['dap']),
            status,
            p['notes']
        ))
        
        # Simpan observasi
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
                str(row.get('growth_stage', row.get('criteria', 'Vegetatif')))
            ))
            
        cursor.executemany("""
        INSERT INTO observations (plant_id, day, dap, time, soil_moisture, temperature, soil_condition, maturity_level, criteria)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, obs_records)
    
    # Anomali awal berbasis data aktual
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
    print("Database populate completed successfully.")
