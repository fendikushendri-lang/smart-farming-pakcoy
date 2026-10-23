import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from services.data_service import get_plant_details, get_growth_history, get_latest_observation
from services.lstm_service import predict_growth_4days
from services.anomaly_service import detect_anomalies_for_plant
from web.sensor_panels import (
    render_simulation_controller,
    render_soil_moisture_sensor_panel,
    render_temperature_sensor_panel,
    handle_simulation_playback
)

def render_plant_detail(selected_plant_id: str):
    plant = get_plant_details(selected_plant_id)
    if not plant:
        st.error(f"Data tanaman dengan ID {selected_plant_id} tidak ditemukan.")
        return
        
    history_df = get_growth_history(selected_plant_id, limit=200)
    anomalies = detect_anomalies_for_plant(selected_plant_id)
    forecast_data = predict_growth_4days(selected_plant_id)
    
    if len(history_df) == 0:
        st.warning("Data observasi belum tersedia untuk tanaman ini.")
        return
    
    # --- 1. HERO BANNER ---
    st.html(f"""
    <div class="hero-container" style="background: linear-gradient(135deg, #064e3b 0%, #059669 45%, #0284c7 100%);">
        <div class="hero-badge">🌿 Profil Tanaman Terfokus</div>
        <h1 class="hero-title">{plant['name']} ({plant['plant_id']})</h1>
        <p class="hero-subtitle">
            Batch: <strong>{plant['batch']}</strong> | Varietas: <strong>{plant['variety']}</strong> | 
            Tgl Tanam: <strong>{plant['planting_date']}</strong> | Status: <strong>{plant['status']}</strong>
        </p>
    </div>
    """)
    
    # --- 1.5 SIMULASI WAKTU MAJU (PLAY/PAUSE TIME-LAPSE CONTROLLER) ---
    sim_obs, sim_idx = render_simulation_controller(history_df, selected_plant_id, key_prefix="pdetail")
    latest = sim_obs
    
    # --- 2. 4 METRIC CARDS MENGIKUTI SIMULASI ---
    dap_val = int(latest['dap'])
    if dap_val >= 40:
        dap_badge = "kpi-delta-warn"
        dap_status = "🚨 Batas Panen 40 HST!"
    elif dap_val >= 35:
        dap_badge = "kpi-delta-up"
        dap_status = f"🥬 Siap Panen (Sisa {40 - dap_val} Hari)"
    else:
        dap_badge = "kpi-delta-info"
        dap_status = f"🌱 Sisa {40 - dap_val} Hari ke Batas Panen"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.html(f"""
        <div class="kpi-card kpi-green">
            <div class="kpi-label">🌱 Umur / Batas Panen</div>
            <div class="kpi-value kpi-value-green">{dap_val} / 40 HST</div>
            <div class="kpi-delta {dap_badge}">{dap_status}</div>
        </div>
        """)
    with c2:
        st.html(f"""
        <div class="kpi-card kpi-purple">
            <div class="kpi-label">📊 Indeks Kematangan</div>
            <div class="kpi-value kpi-value-purple">{latest['maturity_level']:.1f}%</div>
            <div class="kpi-delta kpi-delta-purple">{latest['criteria'].split(' (')[0]}</div>
        </div>
        """)
    with c3:
        st.html(f"""
        <div class="kpi-card kpi-blue">
            <div class="kpi-label">💧 Kelembaban Tanah</div>
            <div class="kpi-value kpi-value-blue">{latest['soil_moisture']}%</div>
            <div class="kpi-delta kpi-delta-info">Kondisi: {latest['soil_condition']}</div>
        </div>
        """)
    with c4:
        st.html(f"""
        <div class="kpi-card kpi-amber">
            <div class="kpi-label">🌡️ Suhu Lingkungan</div>
            <div class="kpi-value kpi-value-amber">{latest['temperature']}°C</div>
            <div class="kpi-delta kpi-delta-up">Kisaran 22-28°C Optimal</div>
        </div>
        """)
        
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # --- 3. PANEL SENSOR KELEMBAPAN TANAH & SUHU (REQUEST USER) ---
    render_soil_moisture_sensor_panel(history_df, sim_idx)
    render_temperature_sensor_panel(history_df, sim_idx)
    
    # --- 3. TIMELINE PERTUMBUHAN INTERAKTIF DENGAN MILESTONE PANEN 40 HST ---
    st.markdown("<h3 style='color: #065f46;'>📅 Timeline Tahapan Pertumbuhan Tanaman (Maksimal 40 HST)</h3>", unsafe_allow_html=True)
    st.caption("Alur kronologis pengamatan dari fase semai awal, kondisi saat ini, hingga proyeksi peramalan 4 hari ke depan dengan batas akhir 40 HST:")
    
    timeline_days = history_df['day'].unique()
    sample_days = [d for d in timeline_days if d % 6 == 0 or d == timeline_days[0] or d == timeline_days[-1]]
    if len(sample_days) > 5:
        sample_days = sample_days[-5:]
        
    t_cols = st.columns(len(sample_days) + 4)
    col_idx = 0
    for d in sample_days:
        d_row = history_df[history_df['day'] == d].iloc[-1]
        is_latest = (d == latest['day'])
        border_style = "2px solid #059669; background: #ecfdf5;" if is_latest else "1px solid #d1fae5; background: #ffffff;"
        with t_cols[col_idx]:
            st.html(f"""
            <div class="timeline-node" style="{border_style}">
                <div style="font-size: 0.72rem; color: #047857; font-weight: 800; text-transform: uppercase;">
                    {'📍 HARI INI' if is_latest else f'HARI {d}'}
                </div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #065f46; margin: 4px 0;">
                    {d_row['maturity_level']}%
                </div>
                <div style="font-size: 0.7rem; color: #64748b;">
                    DAP {d_row['dap']} HST
                </div>
            </div>
            """)
        col_idx += 1
        
    # Projected future days with harvest boundary tracking
    if forecast_data:
        for day_off in range(1, 5):
            sub_f = forecast_data["detailed_forecast_df"][forecast_data["detailed_forecast_df"]["day_label"] == f"Hari +{day_off}"]
            pred_mat = sub_f['maturity_level'].iloc[-1]
            proj_dap = dap_val + day_off
            
            if proj_dap == 40:
                future_style = "border: 2px solid #ef4444; background: #fef2f2;"
                f_title_color = "#b91c1c"
                f_title = "🏁 BATAS 40 HST"
                f_sub = "Batas Panen Mutlak"
            elif proj_dap > 40:
                future_style = "border: 2px solid #f59e0b; background: #fffbeb;"
                f_title_color = "#b45309"
                f_title = f"⚠️ H+{day_off} (>40 HST)"
                f_sub = f"Lewat {proj_dap - 40} Hari"
            else:
                future_style = "border: 1px dashed #38bdf8; background: #f0f9ff;"
                f_title_color = "#0284c7"
                f_title = f"🔮 HARI +{day_off}"
                f_sub = f"{proj_dap} HST (LSTM)"
                
            with t_cols[col_idx]:
                st.html(f"""
                <div class="timeline-node" style="{future_style}">
                    <div style="font-size: 0.72rem; color: {f_title_color}; font-weight: 800; text-transform: uppercase;">
                        {f_title}
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #0369a1; margin: 4px 0;">
                        {pred_mat:.1f}%
                    </div>
                    <div style="font-size: 0.7rem; color: #64748b;">
                        {f_sub}
                    </div>
                </div>
                """)
            col_idx += 1
            
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # --- 4. GRAFIK PERTUMBUHAN AKTUAL VS PROYEKSI PREDIKSI ---
    st.markdown("<h3 style='color: #065f46;'>📈 Kurva Pertumbuhan Historis vs Proyeksi Model LSTM</h3>", unsafe_allow_html=True)
    
    hist_days = history_df['day'].tolist()
    hist_mat = history_df['maturity_level'].tolist()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_days,
        y=hist_mat,
        mode="lines+markers",
        name="Data Historis Aktual (%)",
        line=dict(color="#059669", width=3, shape="spline"),
        marker=dict(size=5, color="#047857"),
        fill="tozeroy",
        fillcolor="rgba(16, 185, 129, 0.12)"
    ))
    
    if forecast_data:
        pred_df = forecast_data["detailed_forecast_df"]
        pred_daily = pred_df.groupby('day')['maturity_level'].last().reset_index()
        proj_days = [hist_days[-1]] + pred_daily['day'].tolist()
        proj_mat = [hist_mat[-1]] + pred_daily['maturity_level'].tolist()
        
        fig.add_trace(go.Scatter(
            x=proj_days,
            y=proj_mat,
            mode="lines+markers",
            name="Proyeksi Prediksi 4 Hari (%)",
            line=dict(color="#0284c7", width=3, dash="dash"),
            marker=dict(size=8, symbol="star", color="#0369a1")
        ))
        
    fig.update_layout(
        title=f"Kurva Pertumbuhan & Horizon Prediksi: {selected_plant_id}",
        xaxis_title="Hari Pengamatan (Day)",
        yaxis_title="Tingkat Kematangan (%)",
        hovermode="x unified",
        template="plotly_white",
        plot_bgcolor="rgba(248, 250, 252, 0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- 5. RINGKASAN ANALISIS AI TERSTRUKTUR ---
    st.markdown("<h3 style='color: #065f46; margin-top: 18px;'>🤖 Analisis Cerdas & Rekomendasi Terpadu</h3>", unsafe_allow_html=True)
    
    c_ai1, c_ai2 = st.columns([1, 1])
    with c_ai1:
        st.html(f"""
        <div class="plant-card" style="border-top: 4px solid #059669; background: #ffffff;">
            <h4 style="color: #065f46; margin-top: 0; display: flex; align-items: center; gap: 6px;">
                📌 Temuan Berdasarkan Data Aktual
            </h4>
            <div style="font-size: 0.9rem; line-height: 1.7; color: #334155;">
                <p>• <strong>Kondisi Fisiologis:</strong> Tanaman berada pada fase <em>{latest['criteria']}</em> dengan tingkat kematangan <strong>{latest['maturity_level']}%</strong>.</p>
                <p>• <strong>Siklus Panen:</strong> Berumur <strong>{dap_val} HST</strong> dari batas maksimal <strong>40 HST</strong> ({'🚨 Wajib panen hari ini!' if dap_val >= 40 else f'Sisa {40 - dap_val} hari'}).</p>
                <p>• <strong>Status Kelembaban:</strong> Sensor media tanam mencatat <strong>{latest['soil_moisture']}%</strong> ({latest['soil_condition']}).</p>
                <p>• <strong>Suhu Lingkungan:</strong> Rata-rata mikro berada di angka <strong>{latest['temperature']}°C</strong>.</p>
                <p>• <strong>Catatan Anomali:</strong> Terdeteksi {len(anomalies)} peringatan parameter lingkungan.</p>
            </div>
        </div>
        """)
        
    with c_ai2:
        harvest_target_rec = ('⚠️ <strong>SEGERA PANEN:</strong> Tanaman telah mencapai batas maksimal 40 HST. Panen pagi ini sebelum pukul 09.00 untuk menjaga kerenyahan pelepah daun.' if dap_val >= 40 else (f'🥬 <strong>JENDELA PANEN PRIMA:</strong> Jadwalkan pemanenan dalam {40 - dap_val} hari ke depan sebelum melewati batas 40 HST.' if dap_val >= 35 else f'🌱 <strong>PEMELIHARAAN VEGETATIF:</strong> Lanjutkan fertigasi optimal, panen dijadwalkan {40 - dap_val} hari lagi.'))
        st.html(f"""
        <div class="plant-card" style="border-top: 4px solid #0284c7; background: #ffffff;">
            <h4 style="color: #0369a1; margin-top: 0; display: flex; align-items: center; gap: 6px;">
                🌱 Rekomendasi Tindakan Agronomi
            </h4>
            <div style="font-size: 0.9rem; line-height: 1.7; color: #334155;">
                <p>• <strong>Batas Panen (40 HST):</strong> {harvest_target_rec}</p>
                <p>• <strong>Pengaturan Irigasi:</strong> Pertahankan kelembaban tanah di kisaran ideal <strong>50% - 60%</strong>.</p>
                <p>• <strong>Sirkulasi & Ventilasi:</strong> Pastikan aerasi optimal saat suhu siang mendekati batas atas 28-29°C.</p>
                <p>• <strong>Inspeksi Visual:</strong> Ambil foto berkala untuk verifikasi kanopi daun pada modul CNN.</p>
            </div>
        </div>
        """)

    # --- ADVANCE SIMULATION PLAYBACK LOOP IF PLAYING ---
    handle_simulation_playback(history_df)
