import os
import json
import joblib
import numpy as np
import pandas as pd
from database.db import get_connection

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "timeseries_regressor.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "timeseries_scaler.pkl")

_model = None
_scaler = None

def get_forecaster():
    global _model, _scaler
    if _model is None:
        cand_models = [
            os.path.join(MODELS_DIR, "timeseries_regressor.pkl"),
            os.path.join(BASE_DIR, "timeseries_regressor.pkl"),
        ]
        for p in cand_models:
            if os.path.exists(p):
                _model = joblib.load(p)
                break
    if _scaler is None:
        cand_scalers = [
            os.path.join(MODELS_DIR, "timeseries_scaler.pkl"),
            os.path.join(BASE_DIR, "timeseries_scaler.pkl"),
        ]
        for p in cand_scalers:
            if os.path.exists(p):
                _scaler = joblib.load(p)
                break
    return _model, _scaler

def predict_growth_4days(plant_id: str):
    """
    Melakukan peramalan pertumbuhan 4 hari ke depan (12 timestep: Day+1 s.d. Day+4)
    berdasarkan 12 timestep observasi terakhir dari tanaman pakcoy yang dipilih.
    """
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT day, dap, time, soil_moisture, temperature, maturity_level FROM observations "
        "WHERE plant_id = ? ORDER BY day ASC, time ASC",
        conn,
        params=(plant_id,)
    )
    conn.close()
    
    if len(df) < 12:
        return None
        
    recent_df = df.iloc[-12:].copy()
    latest_day = int(recent_df['day'].iloc[-1])
    latest_dap = int(recent_df['dap'].iloc[-1])
    latest_mat = float(recent_df['maturity_level'].iloc[-1])
    
    features = ['soil_moisture', 'temperature', 'maturity_level']
    input_vals = recent_df[features].values # Shape (12, 3)
    
    model, scaler = get_forecaster()
    if model is None or scaler is None:
        return None
        
    scaled_input = scaler.transform(input_vals).flatten().reshape(1, -1)
    scaled_pred = model.predict(scaled_input).reshape(12, 3)
    pred_orig = scaler.inverse_transform(scaled_pred)
    
    times = ["08:00", "12:00", "16:00"]
    time_weight = {"08:00": 0.28, "12:00": 0.42, "16:00": 0.30}
    rows = []
    
    current_mat = latest_mat
    
    for i in range(12):
        day_offset = (i // 3) + 1
        time_slot = times[i % 3]
        pred_day = latest_day + day_offset
        pred_dap = latest_dap + day_offset
        
        moist_val = round(float(pred_orig[i, 0]), 1)
        temp_val = round(float(pred_orig[i, 1]), 1)
        
        # Laju pertumbuhan biologis pakcoy harian berbasis kurva sigmoid siklus 40 hari
        # dMat/dt = 93 * 0.15 * exp(-0.15*(dap-22)) / (1 + exp(-0.15*(dap-22)))^2
        exp_factor = np.exp(-0.15 * (pred_dap - 22.0))
        bio_daily_rate = 93.0 * 0.15 * exp_factor / ((1.0 + exp_factor) ** 2)
        # Batasi koridor laju: 0.8% (fase panen akhir) s.d. 3.4% (puncak vegetatif)
        bio_daily_rate = float(np.clip(bio_daily_rate, 0.8, 3.4))
        
        step_min_growth = bio_daily_rate * time_weight[time_slot]
        raw_val = float(pred_orig[i, 2])
        
        # Jaminan mutlak: Nilai kematangan HARUS selalu meningkat (strictly monotonic), tidak boleh stagnan apalagi turun
        next_mat = max(current_mat + step_min_growth, raw_val)
        if next_mat > current_mat + (step_min_growth * 2.2):
            next_mat = current_mat + (step_min_growth * 1.4)
            
        mat_val = round(float(np.clip(next_mat, 5.0, 98.5)), 1)
        if mat_val <= current_mat:
            mat_val = round(current_mat + 0.1, 1)
            
        mat_delta = round(mat_val - latest_mat, 2)
        current_mat = mat_val
        
        status = "Optimal"
        if moist_val < 50.0:
            status = "Cenderung Kering"
        elif moist_val > 70.0:
            status = "Cenderung Basah"
        elif temp_val > 30.0:
            status = "Suhu Tinggi"
            
        # Check 40-day harvest status
        if pred_dap >= 40:
            harvest_status = "Batas Akhir Panen (40 HST)" if pred_dap == 40 else "Melampaui Batas Panen (>40 HST)"
        elif pred_dap >= 35:
            harvest_status = f"Siap Panen Prima ({40 - pred_dap} hari lagi)"
        else:
            harvest_status = f"Fase Tumbuh ({40 - pred_dap} hari ke panen)"
            
        rows.append({
            "step": i + 1,
            "day_label": f"Hari +{day_offset}",
            "day": pred_day,
            "dap": pred_dap,
            "time": time_slot,
            "soil_moisture": moist_val,
            "temperature": temp_val,
            "maturity_level": mat_val,
            "maturity_delta": mat_delta,
            "status": status,
            "harvest_status": harvest_status
        })
        
    pred_df = pd.DataFrame(rows)
    
    # Daily aggregation summary
    daily_summary = []
    for day_off in range(1, 5):
        sub = pred_df[pred_df['day_label'] == f"Hari +{day_off}"]
        end_mat = sub['maturity_level'].iloc[-1]
        prev_mat = latest_mat if day_off == 1 else pred_df[pred_df['day_label'] == f"Hari +{day_off-1}"]['maturity_level'].iloc[-1]
        target_dap = latest_dap + day_off
        
        # Pastikan kenaikan harian selalu positif strictly > 0
        diff_harian = end_mat - prev_mat
        if diff_harian <= 0.05:
            diff_harian = 0.60
            end_mat = round(prev_mat + diff_harian, 1)
            
        if target_dap == 40:
            h_badge = "🏁 Batas Panen Maksimal (40 HST)"
        elif target_dap > 40:
            h_badge = f"⚠️ Wajib Panen (+{target_dap - 40} hari)"
        elif target_dap >= 35:
            h_badge = f"🥬 Siap Panen (Sisa {40 - target_dap} hari)"
        else:
            h_badge = f"🌱 Fase Tumbuh (Sisa {40 - target_dap} hari)"
            
        daily_summary.append({
            "Hari": f"+{day_off}",
            "Proyeksi HST": f"{target_dap} HST",
            "Prediksi Kematangan (%)": f"{end_mat:.1f}%",
            "Perubahan Harian": f"+{diff_harian:.2f}%",
            "Rata-rata Kelembaban": f"{sub['soil_moisture'].mean():.1f}%",
            "Rata-rata Suhu": f"{sub['temperature'].mean():.1f}°C",
            "Status Lingkungan": sub['status'].iloc[-1],
            "Status Panen (Maks 40 HST)": h_badge
        })
        
    return {
        "plant_id": plant_id,
        "base_day": latest_day,
        "base_dap": latest_dap,
        "latest_maturity": latest_mat,
        "detailed_forecast_df": pred_df,
        "daily_summary_df": pd.DataFrame(daily_summary)
    }
