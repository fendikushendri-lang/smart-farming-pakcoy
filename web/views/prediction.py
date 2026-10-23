import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from services.lstm_service import predict_growth_4days
from services.data_service import get_all_plants, get_latest_observation, get_growth_history

def render_prediction(selected_plant_id: str):
    latest_obs = get_latest_observation(selected_plant_id)
    history_df = get_growth_history(selected_plant_id, limit=30)
    forecast_result = predict_growth_4days(selected_plant_id)
    
    st.html(f"""
    <div class="hero-container" style="background: linear-gradient(135deg, #064e3b 0%, #0284c7 50%, #0ea5e9 100%);">
        <div class="hero-badge">📈 Multivariate LSTM Forecaster</div>
        <h1 class="hero-title">Peramalan Pertumbuhan 4 Hari: {selected_plant_id}</h1>
        <p class="hero-subtitle">
            Model deret waktu memproyeksikan 12 timestep simultan (3 pengamatan/hari pada 08:00, 12:00, 16:00 WIB) 
            untuk parameter Kematangan (%), Kelembaban Tanah (%), dan Suhu (°C).
        </p>
    </div>
    """)
    
    if not forecast_result:
        st.warning("Data observasi tanaman ini memerlukan minimal 12 rekaman data untuk menghasilkan peramalan LSTM 4 hari.")
        return
        
    daily_sum = forecast_result["daily_summary_df"]
    det_df = forecast_result["detailed_forecast_df"]
    
    # --- 1. 4-DAY FORECAST CARDS (VIBRANT & COLORFUL) ---
    st.markdown("<h3 style='color: #065f46; margin-bottom: 14px;'>🔮 Estimasi Horizon 4 Hari ke Depan</h3>", unsafe_allow_html=True)
    
    f_cols = st.columns(4)
    card_styles = [
        {"bg": "kpi-green", "badge": "kpi-delta-up", "val_color": "kpi-value-green", "border": "#10b981"},
        {"bg": "kpi-emerald", "badge": "kpi-delta-up", "val_color": "kpi-value-green", "border": "#059669"},
        {"bg": "kpi-blue", "badge": "kpi-delta-info", "val_color": "kpi-value-blue", "border": "#0284c7"},
        {"bg": "kpi-purple", "badge": "kpi-delta-purple", "val_color": "kpi-value-purple", "border": "#8b5cf6"}
    ]
    
    for i in range(4):
        row = daily_sum.iloc[i]
        c_style = card_styles[i]
        target_hst = row.get('Proyeksi HST', '')
        panen_badge = row.get('Status Panen (Maks 40 HST)', '')
        with f_cols[i]:
            st.html(f"""
            <div class="kpi-card {c_style['bg']}" style="padding: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <span class="kpi-label" style="font-size: 0.82rem; margin: 0;">📅 Hari {row['Hari']}</span>
                    <span style="font-size: 0.76rem; font-weight: 800; color: #047857; background: #ffffff; padding: 1px 6px; border-radius: 4px; border: 1px solid #a7f3d0;">{target_hst}</span>
                </div>
                <div class="kpi-value {c_style['val_color']}">{row['Prediksi Kematangan (%)']}</div>
                <div class="kpi-delta {c_style['badge']}">{row['Perubahan Harian']}</div>
                <div style="font-size: 0.74rem; font-weight: 700; color: #1e293b; background: #ffffff; padding: 4px 8px; border-radius: 6px; margin: 6px 0 4px 0; border: 1px solid #e2e8f0;">
                    {panen_badge}
                </div>
                <div style="font-size: 0.74rem; color: #64748b; border-top: 1px dashed #cbd5e1; padding-top: 5px;">
                    💧 {row['Rata-rata Kelembaban']} | 🌡️ {row['Rata-rata Suhu']}
                </div>
            </div>
            """)
            
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # --- 2. SUMMARY TABLE ---
    st.markdown("<h3 style='color: #065f46;'>📋 Rincian Matriks Harian (Standar Panen Maksimal 40 HST)</h3>", unsafe_allow_html=True)
    st.dataframe(daily_sum, use_container_width=True, hide_index=True)
    
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # --- 3. INTERACTIVE CONNECTED CHART ---
    st.markdown("<h3 style='color: #065f46;'>📊 Visualisasi Kurva Proyeksi 12 Timestep</h3>", unsafe_allow_html=True)
    
    tab_p1, tab_p2 = st.tabs(["🌱 Kurva Kematangan (%)", "📑 Tabel Observasi Per Sesi Waktu (08:00, 12:00, 16:00)"])
    
    with tab_p1:
        hist_recent = history_df.tail(12).copy()
        
        fig = go.Figure()
        # Historical trace
        fig.add_trace(go.Scatter(
            x=[f"H-{12-idx}" for idx in range(len(hist_recent))],
            y=hist_recent['maturity_level'],
            mode="lines+markers",
            name="Historis (12 Sesi Terakhir)",
            line=dict(color="#059669", width=3, shape="spline"),
            marker=dict(size=6, color="#047857"),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.12)"
        ))
        
        # Forecast trace
        forecast_labels = [f"H+{det_df['day_label'].iloc[idx].replace('Hari +', '')} ({det_df['time'].iloc[idx]})" for idx in range(len(det_df))]
        fig.add_trace(go.Scatter(
            x=forecast_labels,
            y=det_df['maturity_level'],
            mode="lines+markers",
            name="Proyeksi LSTM 4 Hari",
            line=dict(color="#0284c7", width=3, dash="dash"),
            marker=dict(symbol="star", size=9, color="#0369a1"),
            fill="tozeroy",
            fillcolor="rgba(2, 132, 199, 0.08)"
        ))
        
        fig.update_layout(
            title=f"Sambungan Data Historis dan Peramalan 4 Hari ({selected_plant_id})",
            xaxis_title="Timeline Observasi",
            yaxis_title="Tingkat Kematangan (%)",
            hovermode="x unified",
            template="plotly_white",
            plot_bgcolor="rgba(248, 250, 252, 0.8)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with tab_p2:
        table_view = det_df[['day_label', 'dap', 'time', 'maturity_level', 'maturity_delta', 'soil_moisture', 'temperature', 'status', 'harvest_status']].copy()
        table_view.columns = ['Label Hari', 'Umur (HST)', 'Waktu', 'Kematangan (%)', 'Delta (%)', 'Kelembaban (%)', 'Suhu (°C)', 'Kondisi', 'Status Panen (Maks 40 HST)']
        st.dataframe(
            table_view,
            use_container_width=True,
            hide_index=True
        )
