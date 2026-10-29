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
        
    daily_sum = forecast_result["daily_summary_df"]
    det_df = forecast_result["detailed_forecast_df"]
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    local_img_dir = os.path.join(BASE_DIR, "data", "forecast_images")

    # --- 1. HEADER SECTION MATCHING USER SCREENSHOT ---
    c_head1, c_head2 = st.columns([3.5, 1])
    with c_head1:
        st.html("""
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 16px;">
            <div style="background: #f5f3ff; border: 1.5px solid #ddd6fe; border-radius: 12px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem;">
                🧠
            </div>
            <div>
                <h3 style="color: #0f172a; font-size: 1.28rem; font-weight: 800; margin: 0 0 2px 0;">
                    Prediksi Pertumbuhan 4 Hari ke Depan
                </h3>
                <p style="color: #64748b; font-size: 0.88rem; margin: 0;">
                    Hasil analisis model LSTM berdasarkan data sensor telemetri deret waktu.
                </p>
            </div>
        </div>
        """)
    with c_head2:
        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Muat Ulang", key="btn_reload_pred", use_container_width=True):
            st.rerun()

    # --- 2. 4-DAY FORECAST CARDS WITH REAL PLANT PHOTOS ---
    f_cols = st.columns(4)
    
    for i in range(4):
        row = daily_sum.iloc[i]
        target_hst_str = str(row.get('Proyeksi HST', ''))
        target_dap = int(''.join(filter(str.isdigit, target_hst_str)) or (latest_obs.get('dap', 30) + i + 1))
        day_label = f"HARI KE-{target_dap}"
        
        mat_str = str(row['Prediksi Kematangan (%)'])
        try:
            mat_num = float(mat_str.replace('%', '').strip())
        except Exception:
            mat_num = 25.0
            
        progress_val = min(max(mat_num, 5.0), 100.0)
        
        # Dynamic phase badge based on actual plant day (HST)
        if target_dap < 30:
            phase_text = "Masa Pertumbuhan"
            badge_bg = "#eff6ff"
            badge_color = "#2563eb"
            badge_border = "#bfdbfe"
        elif target_dap < 36:
            phase_text = "Mendekati Panen"
            badge_bg = "#fef3c7"
            badge_color = "#b45309"
            badge_border = "#fde68a"
        else:
            phase_text = "Siap Panen"
            badge_bg = "#ecfdf5"
            badge_color = "#059669"
            badge_border = "#a7f3d0"

        # Determine dynamic plant image matching the specific HST phase
        phase_key = get_plant_phase_key(target_dap)
        img_src = get_forecast_image_for_dap(target_dap, i)
        
        # Robust fallback: check local forecast_images or raw dataset folders on Cloud
        if not img_src:
            img_candidate = os.path.join(local_img_dir, f"{phase_key}_{i+1}.jpg")
            if os.path.exists(img_candidate):
                import base64
                with open(img_candidate, "rb") as f_img:
                    img_src = f"data:image/jpeg;base64,{base64.b64encode(f_img.read()).decode('utf-8')}"
            else:
                cat_map = {
                    "semai": "Masa_Pertumbuhan",
                    "vegetatif": "Masa_Pertumbuhan",
                    "pra_panen": "Mendekati_Panen",
                    "siap_panen": "Siap_Panen"
                }
                cat_name = cat_map.get(phase_key, "Siap_Panen")
                import glob
                found = glob.glob(os.path.join(BASE_DIR, cat_name, "*.jpg")) or glob.glob(os.path.join(BASE_DIR, "data", "images", cat_name, "*.jpg"))
                if found:
                    idx_pick = (i * 3) % len(found)
                    import base64
                    with open(found[idx_pick], "rb") as f_img:
                        img_src = f"data:image/jpeg;base64,{base64.b64encode(f_img.read()).decode('utf-8')}"
                else:
                    img_src = ""

        with f_cols[i]:
            img_html = f"""<img src="{img_src}" alt="Visualisasi Kondisi" style="width: 100%; height: 135px; object-fit: cover; border-radius: 10px; border: 1px solid #e2e8f0; display: block;" />""" if img_src else """<div style="width: 100%; height: 135px; background: #f1f5f9; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #64748b; font-size: 0.8rem;">🌱 Foto Sampel</div>"""
            
            st.html(f"""
            <div style="background: #ffffff; border: 1.5px solid #d1fae5; border-radius: 16px; padding: 16px 18px; box-shadow: 0 2px 10px rgba(0,0,0,0.03); margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.78rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">
                        {day_label}
                    </span>
                    <span style="background: {badge_bg}; color: {badge_color}; font-size: 0.72rem; font-weight: 800; padding: 3px 10px; border-radius: 9999px; border: 1px solid {badge_border};">
                        {phase_text}
                    </span>
                </div>
                
                <div style="font-size: 2.1rem; font-weight: 800; color: #059669; margin: 4px 0 6px 0; line-height: 1.1;">
                    {mat_str}
                </div>
                
                <div style="background: #dcfce7; border-radius: 9999px; height: 6px; overflow: hidden; margin-bottom: 12px;">
                    <div style="background: #10b981; width: {progress_val}%; height: 100%; border-radius: 9999px;"></div>
                </div>
                
                <div style="font-size: 0.76rem; font-weight: 600; color: #64748b; margin-bottom: 6px;">
                    Visualisasi Kondisi:
                </div>
                
                {img_html}
            </div>
            """)
            
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # --- 3. SUMMARY TABLE ---
    st.markdown("<h3 style='color: #065f46;'>📋 Rincian Matriks Harian (Standar Panen Maksimal 40 HST)</h3>", unsafe_allow_html=True)
    st.dataframe(daily_sum, use_container_width=True, hide_index=True)
    
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # --- 4. INTERACTIVE CONNECTED CHART ---
    st.markdown("<h3 style='color: #065f46;'>📊 Visualisasi Kurva Proyeksi 12 Timestep</h3>", unsafe_allow_html=True)
    
    tab_p1, tab_p2 = st.tabs(["🌱 Kurva Kematangan (%)", "📑 Tabel Observasi Per Sesi Waktu (08:00, 12:00, 16:00)"])
    
    with tab_p1:
        # Connected History + Forecast Plot
        fig_pred = go.Figure()
        
        # Historical actual trace
        recent_hist = history_df.tail(15)
        fig_pred.add_trace(go.Scatter(
            x=[f"H{r['day']} {r['time']}" for _, r in recent_hist.iterrows()],
            y=recent_hist['maturity_level'],
            mode="lines+markers",
            name="Historis Sensorik Aktual",
            line=dict(color="#047857", width=2.5),
            marker=dict(size=6, color="#065f46")
        ))
        
        # Forecast trace
        forecast_x = [f"{r.get('day_label', 'Hari +' + str(r.get('step', '')))} {r.get('time', '')}" for _, r in det_df.iterrows()]
        mat_series = det_df['maturity_level'] if 'maturity_level' in det_df.columns else (det_df['predicted_maturity'] if 'predicted_maturity' in det_df.columns else [35.0] * len(det_df))
        fig_pred.add_trace(go.Scatter(
            x=forecast_x,
            y=mat_series,
            mode="lines+markers",
            name="Proyeksi Model LSTM (4 Hari)",
            line=dict(color="#0284c7", width=3, dash="dash"),
            marker=dict(size=7, color="#0369a1", symbol="diamond")
        ))
        
        # Maximum 40-day harvest threshold reference line
        fig_pred.add_hline(
            y=37.0,
            line_dash="dot",
            line_color="#10b981",
            line_width=2,
            annotation_text="Ambang Batas Kematangan Panen Prima (37%)",
            annotation_position="top left"
        )
        
        fig_pred.update_layout(
            title=f"Koneksi Kurva Historis vs Proyeksi Model LSTM: {selected_plant_id}",
            xaxis_title="Kronologi Waktu (Historis → 4 Hari ke Depan)",
            yaxis_title="Kematangan (%)",
            hovermode="x unified",
            template="plotly_white",
            plot_bgcolor="rgba(248, 250, 252, 0.8)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        
    with tab_p2:
        renamed_det = {
            "step": "Timestep",
            "day_label": "Proyeksi Hari",
            "day": "Hari Ke-",
            "dap": "HST",
            "time": "Waktu",
            "soil_moisture": "Kelembaban (%)",
            "temperature": "Suhu (°C)",
            "maturity_level": "Prediksi Kematangan (%)",
            "maturity_delta": "Perubahan (%)",
            "status": "Kondisi Lingkungan",
            "harvest_status": "Status Panen"
        }
        st.dataframe(det_df.rename(columns=renamed_det), use_container_width=True, hide_index=True)
