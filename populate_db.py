import os
import shutil
import numpy as np
import pandas as pd
from database.db import get_connection, init_db, DB_PATH

def generate_telemetry_dataset():
    """Menghasilkan dataset telemetri sensorik IoT 40 hari penuh (1.600 observasi)."""
    np.random.seed(42)
    records = []
    
    total_days = 40
    readings_per_day = 40
    
    minutes_list = np.linspace(6 * 60, 22.5 * 60, readings_per_day, dtype=int)
    time_strings = [f"{m // 60:02d}:{m % 60:02d}" for m in minutes_list]
    
    current_moisture = 58.5
    pump_cooldown = 0
    
    for day in range(1, total_days + 1):
        dap = day  # 1 HST s.d. 40 HST
        
        leaf_width_base = 1.8 + (10.0 / (1.0 + np.exp(-0.16 * (day - 22))))
        maturity_base = 5.0 + (93.0 / (1.0 + np.exp(-0.15 * (day - 22))))
        
        for step, t_str in enumerate(time_strings):
            hour = int(t_str.split(":")[0])
            minute = int(t_str.split(":")[1])
            
            diurnal = np.sin((hour + minute / 60.0 - 8.0) * np.pi / 12.0)
            temp = 25.5 + 3.0 * diurnal + np.random.normal(0, 0.25)
            temp = round(float(np.clip(temp, 22.0, 29.5)), 1)
            
            evaporation_rate = 0.35 if (10 <= hour <= 15) else 0.18
            current_moisture -= evaporation_rate + np.random.uniform(0.02, 0.12)
            
            if current_moisture < 50.0 or pump_cooldown > 0:
                if pump_cooldown == 0:
                    pump_cooldown = 2
                pump_status = "ON"
                current_moisture += np.random.uniform(4.5, 6.5)
                pump_cooldown -= 1
            else:
                pump_status = "OFF"
                
            current_moisture = round(float(np.clip(current_moisture, 45.0, 75.0)), 1)
            
            if current_moisture < 50.0:
                soil_cond = "Cenderung Kering"
            elif current_moisture > 68.0:
                soil_cond = "Cenderung Basah"
            else:
                soil_cond = "Optimal"
                
            leaf_w = round(float(np.clip(leaf_width_base + np.random.normal(0, 0.12), 1.5, 12.5)), 1)
            
            step_progress = (day - 1) + (step / readings_per_day)
            mat_val = 5.0 + (93.0 / (1.0 + np.exp(-0.15 * (step_progress - 22.0))))
            mat_val = round(float(np.clip(mat_val, 5.0, 98.0)), 1)
            
            if dap < 17:
                crit = "Masa Pertumbuhan"
            elif dap < 29:
                crit = "Vegetatif"
            elif dap < 36:
                crit = "Mendekati Panen"
            else:
                crit = "Siap Panen"
                
            records.append({
                "day": day,
                "dap": dap,
                "time": t_str,
                "soil_moisture": current_moisture,
                "temperature": temp,
                "soil_condition": soil_cond,
                "pump_status": pump_status,
                "leaf_width": leaf_w,
                "maturity_level": mat_val,
                "criteria": crit,
                "growth_stage": crit,
                "timestep": len(records) + 1
            })
            
    return pd.DataFrame(records)

def populate(force=False):
    """
    Mengisi database SQLite pakcoy_smart_farming.db dengan urutan kronologis yang benar:
    - PCK-001: 10 HST (Semai)
    - PCK-002: 20 HST (Vegetatif)
    - PCK-003: 30 HST (Pre-Harvest)
    - PCK-004: 36 HST (Siap Panen)
    - PCK-005: 40 HST (Batas Panen Maksimal)
    - PCK-FULL40: 40 HST (Siklus Lengkap 40 Hari)
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    # Deteksi otomatis apakah urutan tanaman sudah benar dan lengkap
    needs_repopulate = force
    try:
        cursor.execute("SELECT plant_id, current_dap FROM plants WHERE plant_id IN ('PCK-001', 'PCK-002', 'PCK-003', 'PCK-004', 'PCK-005')")
        p_dict = dict(cursor.fetchall())
        if (p_dict.get('PCK-001') != 10 or 
            p_dict.get('PCK-002') != 20 or 
            p_dict.get('PCK-003') != 30 or 
            p_dict.get('PCK-004') != 36 or 
            p_dict.get('PCK-005') != 40):
            needs_repopulate = True
            print("Urutan kronologis tanaman versi lama terdeteksi tidak sesuai. Mereset otomatis...")
    except Exception:
        needs_repopulate = True

    try:
        cursor.execute("SELECT COUNT(*) FROM observations")
        cnt = cursor.fetchone()[0]
        if cnt < 1600:
            needs_repopulate = True
    except Exception:
        needs_repopulate = True

    if not needs_repopulate:
        conn.close()
        return

    print("Melakukan sinkronisasi & populasi database urutan kronologis PCK-001 (10 HST) s.d. PCK-005 (40 HST)...")
    
    # 1. Bersihkan tabel lama
    cursor.execute("DROP TABLE IF EXISTS observations")
    cursor.execute("""
    CREATE TABLE observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plant_id TEXT NOT NULL,
        day INTEGER NOT NULL,
        dap INTEGER NOT NULL,
        time TEXT NOT NULL,
        soil_moisture REAL NOT NULL,
        temperature REAL NOT NULL,
        soil_condition TEXT NOT NULL,
        pump_status TEXT DEFAULT 'OFF',
        leaf_width REAL DEFAULT 5.0,
        maturity_level REAL NOT NULL,
        criteria TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(plant_id) REFERENCES plants(plant_id)
    )
    """)
    
    cursor.execute("DROP TABLE IF EXISTS plants")
    cursor.execute("""
    CREATE TABLE plants (
        plant_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        batch TEXT NOT NULL,
        variety TEXT DEFAULT 'Pakcoy Green Fortune',
        planting_date TEXT,
        current_day INTEGER DEFAULT 1,
        current_dap INTEGER DEFAULT 1,
        status TEXT DEFAULT 'Optimal',
        notes TEXT
    )
    """)
    
    cursor.execute("DELETE FROM alerts")
    conn.commit()

    # 2. Hasilkan dataset telemetri
    df = generate_telemetry_dataset()
    
    # 3. Konfigurasi tanaman/batch 40 HST (Urutan Kronologis Resmi)
    plants = [
        ("PCK-001", "Pakcoy Semai Polybag A-1 (Semai 10 HST)", "Batch-2026-SEM-01", "Pakcoy Green Fortune", "2026-10-20", 10, 10, "Optimal", "Fase semai dan persemaian awal (10 HST). Daun kotiledon tumbuh seragam 2-4 helai."),
        ("PCK-002", "Pakcoy Polybag B-2 (Vegetatif 20 HST)", "Batch-2026-SUB-02", "Pakcoy Nauli F1", "2026-10-10", 20, 20, "Optimal", "Fase vegetatif aktif (20 HST) lebar daun 5.8 cm, irigasi pompa otomatis terjaga."),
        ("PCK-003", "Pakcoy DWC C-1 (Pre-Harvest 30 HST)", "Batch-2026-DWC-03", "Pakcoy Green Fortune", "2026-09-30", 30, 30, "Optimal", "Fase mendekati panen (30 HST), tajuk roset daun rimbun 8.5 cm."),
        ("PCK-004", "Pakcoy Vertikal Tower D-2 (Siap Panen 36 HST)", "Batch-2026-AERO-04", "Pakcoy Nauli F1", "2026-09-24", 36, 36, "Harvest Ready", "Fase siap panen komersial optimal (36 HST), lebar daun 10.5 cm siap petik prima."),
        ("PCK-005", "Pakcoy Smart IoT E-1 (Batas Panen 40 HST)", "Batch-2026-IoT-05", "Pakcoy Green Fortune", "2026-09-20", 40, 40, "Harvest Ready", "Siklus batas panen maksimal pakcoy 40 HST, lebar daun 11.2 cm (Standar pasar komersial terpenuhi)."),
        ("PCK-FULL40", "Pakcoy Siklus Lengkap 40 Hari", "Batch-2026-FULL-40D", "Pakcoy Green Fortune", "2026-09-20", 40, 40, "Harvest Ready", "Siklus pengamatan kontinu 40 hari penuh (1.600 observasi lengkap, Day 1-40) mencakup seluruh tahapan budidaya."),
    ]
    cursor.executemany("""
    INSERT INTO plants (plant_id, name, batch, variety, planting_date, current_day, current_dap, status, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, plants)
    
    # 4. Simpan rekaman observasi per tanaman
    plant_ranges = [
        ("PCK-001", 10),
        ("PCK-002", 20),
        ("PCK-003", 30),
        ("PCK-004", 36),
        ("PCK-005", 40),
        ("PCK-FULL40", 40),
    ]
    
    for target_pid, end_d in plant_ranges:
        sub_records = []
        sub_df = df[df['day'] <= end_d]
        for _, r in sub_df.iterrows():
            sub_records.append((
                target_pid, int(r['day']), int(r['dap']), r['time'],
                float(r['soil_moisture']), float(r['temperature']), r['soil_condition'],
                r['pump_status'], float(r['leaf_width']), float(r['maturity_level']), r['criteria']
            ))
        cursor.executemany("""
        INSERT INTO observations (plant_id, day, dap, time, soil_moisture, temperature, soil_condition, pump_status, leaf_width, maturity_level, criteria)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sub_records)
        
    # 5. Notifikasi & Anomali Peringatan Cerdas
    anomalies = [
        ("PCK-001", 3, "Soil Condition", "Medium", "Kelembaban Tinggi (Wet)", "Kelembaban media tanam semai tercatat 72% pada pagi hari, sirkulasi udara ditingkatkan."),
        ("PCK-003", 26, "Soil Condition", "High", "Kekeringan Media Tanam (Dry)", "Kelembaban media tanam tercatat 48%, pompa penyiraman otomatis langsung diaktifkan."),
        ("PCK-005", 40, "Masa Panen", "High", "Batas Panen Maksimal (40 HST)", "Tanaman PCK-005 telah mencapai batas usia panen maksimal pakcoy 40 HST (Standar kualitas pasar komersial). Segera lakukan pemanenan."),
    ]
    for a in anomalies:
        cursor.execute("""
        INSERT INTO alerts (plant_id, day, alert_type, severity, title, description, status)
        VALUES (?, ?, ?, ?, ?, ?, 'Active')
        """, a)

    conn.commit()
    conn.close()
    
    # Sinkronisasi ke root jika file DB root ada
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        root_db_file = os.path.join(BASE_DIR, "pakcoy_smart_farming.db")
        if os.path.abspath(DB_PATH) != os.path.abspath(root_db_file):
            shutil.copy2(DB_PATH, root_db_file)
            print("Berhasil menyinkronkan pakcoy_smart_farming.db ke root direktori.")
    except Exception as e:
        print(f"Catatan sinkronisasi root DB: {e}")
        
    print("Populasi database dengan urutan kronologis PCK-001 s.d. PCK-005 sukses!")

if __name__ == "__main__":
    populate(force=True)
