import os
import sqlite3
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "pakcoy_smart_farming.db")
CSV_PATH_1 = os.path.join(BASE_DIR, "data", "processed", "timeseries_clean.csv")
CSV_PATH_2 = os.path.join(BASE_DIR, "data", "timeseries_clean.csv")

def generate_telemetry_dataset():
    np.random.seed(42)
    records = []
    
    # 40 Hari siklus tanam pakcoy (Day 1 s.d. Day 40, DAP 1 s.d. 40 HST)
    # 40 Pembacaan per hari = TEPAT 1.600 rekaman data!
    total_days = 40
    readings_per_day = 40
    
    # Generate 40 time intervals from 06:00 to 22:15 (setiap 25 menit)
    minutes_list = np.linspace(6 * 60, 22.5 * 60, readings_per_day, dtype=int)
    time_strings = [f"{m // 60:02d}:{m % 60:02d}" for m in minutes_list]
    
    current_moisture = 58.5
    pump_cooldown = 0
    
    for day in range(1, total_days + 1):
        dap = day  # 1 HST s.d. 40 HST
        
        # Pertumbuhan biologis lebar daun (cm) - Sigmoid S-curve
        # Day 1: ~2.0 cm -> Day 20: ~5.8 cm -> Day 35: ~9.8 cm -> Day 40: ~11.2 cm
        leaf_width_base = 1.8 + (10.0 / (1.0 + np.exp(-0.16 * (day - 22))))
        
        # Kematangan (%)
        maturity_base = 5.0 + (93.0 / (1.0 + np.exp(-0.15 * (day - 22))))
        
        for step, t_str in enumerate(time_strings):
            hour = int(t_str.split(":")[0])
            minute = int(t_str.split(":")[1])
            
            # Dinamika Suhu Lingkungan (°C)
            # Puncak siang jam 12:00-14:00 (28.5°C), pagi/malam sejuk (22.5°C - 24.0°C)
            diurnal = np.sin((hour + minute / 60.0 - 8.0) * np.pi / 12.0)
            temp = 25.5 + 3.0 * diurnal + np.random.normal(0, 0.25)
            temp = round(float(np.clip(temp, 22.0, 29.5)), 1)
            
            # Evaporasi dan Dinamika Kelembaban Tanah (%)
            # Siang hari evaporasi lebih cepat
            evaporation_rate = 0.35 if (10 <= hour <= 15) else 0.18
            current_moisture -= evaporation_rate + np.random.uniform(0.02, 0.12)
            
            # Logika Otomasi Pompa Real-Time:
            # Jika kelembaban di bawah 50.0% -> POMPA AKTIF (Menyiram)
            if current_moisture < 50.0 or pump_cooldown > 0:
                if pump_cooldown == 0:
                    pump_cooldown = 2  # Pompa menyiram selama 2 sesi
                pump_status = "ON"
                current_moisture += np.random.uniform(4.5, 6.5)
                pump_cooldown -= 1
            else:
                pump_status = "OFF"
                
            current_moisture = round(float(np.clip(current_moisture, 45.0, 68.0)), 1)
            
            # Status media tanah
            if current_moisture < 50.0:
                soil_cond = "Dry"
            elif current_moisture > 65.0:
                soil_cond = "Wet"
            else:
                soil_cond = "Optimal"
                
            # Lebar daun terukur (dengan sedikit variasi sensorik)
            leaf_w = round(float(leaf_width_base + np.random.normal(0, 0.08)), 2)
            mat_val = round(float(np.clip(maturity_base + np.random.normal(0, 0.15), 5.0, 99.0)), 1)
            
            # Kriteria Kematangan Berdasarkan Lebar Daun (Sesuai Permintaan CNN):
            if leaf_w < 4.5:
                criteria = "Fase Pertumbuhan (Kotiledon / Semai)"
            elif leaf_w < 7.5:
                criteria = "Fase Pertumbuhan (Vegetatif Aktif)"
            elif leaf_w < 10.0:
                criteria = "Fase Mendekati Panen (Pra-Panen)"
            else:
                criteria = "Fase Siap Panen (Matang Optimal - Lebar Daun ≥ 10 cm)"
                
            records.append({
                "plant_id": "PCK-001",
                "day": day,
                "dap": dap,
                "time": t_str,
                "soil_moisture": current_moisture,
                "temperature": temp,
                "soil_condition": soil_cond,
                "pump_status": pump_status,
                "leaf_width": leaf_w,
                "maturity_level": mat_val,
                "criteria": criteria,
                "growth_stage": criteria,
                "timestep": len(records) + 1
            })
            
    df = pd.DataFrame(records)
    print(f"Generated {len(df)} total observation rows across {total_days} days.")
    return df

def save_and_populate_db():
    df = generate_telemetry_dataset()
    
    # 1. Simpan ke CSV
    os.makedirs(os.path.dirname(CSV_PATH_1), exist_ok=True)
    df.to_csv(CSV_PATH_1, index=False)
    df.to_csv(CSV_PATH_2, index=False)
    print(f"Saved CSV dataset to: {CSV_PATH_1} and {CSV_PATH_2}")
    
    # 2. Update Database SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Recreate observations table with pump_status and leaf_width
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
    
    # Recreate plants table
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
    
    # Plants metadata - 40 Batch lengkap dari PCK-001 (1 HST) s.d. PCK-040 (40 HST) & PCK-FULL40
    plants = []
    for day in range(1, 41):
        pid = f"PCK-{day:03d}"
        if day <= 4:
            stage = "Fase Semai"
            status = "Optimal"
            desc = f"Fase semai dan perkecambahan awal ({day} HST). Daun kotiledon tumbuh seragam."
        elif day <= 14:
            stage = "Fase Pembibitan"
            status = "Optimal"
            desc = f"Fase pembibitan/seedling ({day} HST). Daun sejati 2-4 helai, irigasi terjaga optimal."
        elif day <= 29:
            stage = "Fase Vegetatif"
            status = "Optimal"
            desc = f"Fase vegetatif aktif ({day} HST). Pembentukan tajuk roset dan helaian daun hijau segar."
        elif day <= 34:
            stage = "Fase Pra-Panen"
            status = "Optimal"
            desc = f"Fase mendekati panen ({day} HST). Tajuk roset daun rimbun dan lebat."
        elif day <= 39:
            stage = "Fase Siap Panen"
            status = "Harvest Ready"
            desc = f"Fase siap panen komersial prima ({day} HST). Lebar daun >= 10 cm siap petik segar."
        else:
            stage = "Batas Panen Maksimal"
            status = "Harvest Ready"
            desc = f"Batas panen maksimal pakcoy 40 HST. Wajib segera dipanen agar pelepah tidak berserat keras."

        variety = "Pakcoy Green Fortune" if day % 2 == 1 else "Pakcoy Nauli F1"
        name = f"Pakcoy Batch {pid} ({stage} {day} HST)"
        batch_code = f"Batch-2026-PCK-{day:02d}"
        planting_date = f"2026-10-{max(1, 41 - day):02d}"
        plants.append((pid, name, batch_code, variety, planting_date, day, day, status, desc))

    plants.append((
        "PCK-FULL40",
        "Pakcoy Siklus Lengkap 40 Hari",
        "Batch-2026-FULL-40D",
        "Pakcoy Green Fortune",
        "2026-09-20",
        40,
        40,
        "Harvest Ready",
        "Siklus pengamatan kontinu 40 hari penuh (1.600 observasi lengkap, Day 1-40) mencakup seluruh tahapan budidaya."
    ))

    cursor.executemany("""
    INSERT INTO plants (plant_id, name, batch, variety, planting_date, current_day, current_dap, status, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, plants)
    
    # Insert observation records for PCK-001 through PCK-040 and PCK-FULL40
    for day in range(1, 41):
        target_pid = f"PCK-{day:03d}"
        sub_df = df[df['day'] <= day]
        sub_records = []
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

    full_records = []
    for _, r in df.iterrows():
        full_records.append((
            "PCK-FULL40", int(r['day']), int(r['dap']), r['time'],
            float(r['soil_moisture']), float(r['temperature']), r['soil_condition'],
            r['pump_status'], float(r['leaf_width']), float(r['maturity_level']), r['criteria']
        ))
    cursor.executemany("""
    INSERT INTO observations (plant_id, day, dap, time, soil_moisture, temperature, soil_condition, pump_status, leaf_width, maturity_level, criteria)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, full_records)
        
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM observations")
    total_obs = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM observations WHERE plant_id = 'PCK-FULL40'")
    pck_full_obs = cursor.fetchone()[0]
    
    conn.close()
    print(f"Database successfully populated! Total observations: {total_obs}, PCK-FULL40 observations: {pck_full_obs}")

if __name__ == "__main__":
    save_and_populate_db()
