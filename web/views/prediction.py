import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
from services.lstm_service import predict_growth_4days
from services.data_service import get_all_plants, get_latest_observation, get_growth_history
try:
    from data.forecast_b64 import get_forecast_image_for_dap
except Exception:
    def get_forecast_image_for_dap(dap_val: int, step_idx: int) -> str:
        return ""
def get_plant_phase_key(dap_val: int) -> str:
    if dap_val <= 16:
        return "semai"
    elif dap_val <= 28:
        return "vegetatif"
    elif dap_val <= 35:
        return "pra_panen"
    else:
        return "siap_panen"
def render_prediction(selected_plant_id: str):
    latest_obs = get_latest_observation(selected_plant_id)
    history_df = get_growth_history(selected_plant_id, limit=5000)
    forecast_result = predict_growth_4days(selected_plant_id)
    
    st.html(f"""
    <div class="hero-container" style="background: linear-gradient(135deg, #064e3b 0%, #0284c7 50%, #0ea5e9 100%);">
        <div class="hero-badge">📈 Multivariate LSTM Forecaster</div>
        <h1 class="hero-title">Peramalan Pertumbuhan 4 Hari: {selected_plant_id}</h1>
        <p class="hero-subtitle">
            Model deret waktu memproyeksikan dinamika pertumbuhan 4 hari ke depan 
            untuk parameter Kematangan (%), Kelembaban Tanah (%), dan Suhu Lingkungan (°C).
        </p>
    </div>
    """)
    
    if not forecast_result:
        st.warning("Data observasi tanaman ini memerlukan minimal 12 rekaman data untuk menghasilkan peramalan LSTM 4 hari.")
        return
