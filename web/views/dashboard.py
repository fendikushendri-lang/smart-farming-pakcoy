import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from services.data_service import (
    get_all_plants, get_latest_observation, get_growth_history, get_alerts
)
from services.lstm_service import predict_growth_4days
from services.anomaly_service import detect_anomalies_for_plant
from web.sensor_panels import (
    render_simulation_controller,
    render_soil_moisture_sensor_panel,
    render_temperature_sensor_panel,
    handle_simulation_playback
)

def render_dashboard(selected_plant_id: str):
    plants_df = get_all_plants()
    total_plants = len(plants_df)
    history_df = get_growth_history(selected_plant_id, limit=200)
    alerts_df = get_alerts()
    active_anomalies = len(alerts_df)
    
    if len(history_df) == 0:
        st.warning("Data observasi belum tersedia untuk tanaman ini.")
        return

    # --- 1. HERO BANNER ---
    st.html(f"""
    <div class="hero-container">
        <div class="hero-badge">🌱 Smart Pakcoy Monitoring System</div>
        <h1 class="hero-title">Executive Dashboard Pertumbuhan Pakcoy</h1>
        <p class="hero-subtitle">
            Pemantauan presisi berbasis AI cerdas. Menggabungkan penglihatan komputer (CNN) untuk analisis visual daun 
            dan model deret waktu (Multivariate LSTM) untuk memprediksi dinamika pertumbuhan pakcoy 4 hari ke depan.
        </p>
    </div>
    """)
    
    # --- 1.5 SIMULASI WAKTU MAJU (PLAY/PAUSE TIME-LAPSE CONTROLLER) ---
    sim_obs, sim_idx = render_simulation_controller(history_df, selected_plant_id, key_prefix="dash")
    latest_obs = sim_obs
    
    # Calculate delta relative to previous reading
    prev_obs = history_df.iloc[sim_idx - 1] if sim_idx >= 1 else latest_obs
    mat_delta = round(latest_obs['maturity_level'] - prev_obs['maturity_level'], 2)
    delta_symbol = "▲" if mat_delta >= 0 else "▼"
    
    # --- 2. VIBRANT COLORFUL KPI CARDS ---
    st.markdown("<h3 style='color: #065f46; margin-bottom: 14px;'>🌿 Ringkasan Kinerja & Metrik Utama</h3>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.html(f"""
        <div class="kpi-card kpi-green">
            <div class="kpi-label">📊 Tingkat Kematangan</div>
            <div class="kpi-value kpi-value-green">{latest_obs['maturity_level']:.1f}%</div>
            <div class="kpi-delta kpi-delta-up">{delta_symbol} {mat_delta:+.2f}% dari sesi lalu</div>
        </div>
        """)
    with c2:
        moist_color = "kpi-delta-info" if latest_obs['soil_condition'] == 'Optimal' else "kpi-delta-warn"
        st.html(f"""
        <div class="kpi-card kpi-blue">
            <div class="kpi-label">💧 Kelembaban Media</div>
            <div class="kpi-value kpi-value-blue">{latest_obs['soil_moisture']:.1f}%</div>
            <div class="kpi-delta {moist_color}">Kondisi: {latest_obs['soil_condition']}</div>
        </div>
        """)
    with c3:
        temp_val = latest_obs['temperature']
        temp_badge = "kpi-delta-up" if 22 <= temp_val <= 28.5 else "kpi-delta-warn"
        temp_status = "Optimal (22-28°C)" if 22 <= temp_val <= 28.5 else "Perlu Ventilasi"
        st.html(f"""
        <div class="kpi-card kpi-amber">
            <div class="kpi-label">🌡️ Suhu Lingkungan</div>
            <div class="kpi-value kpi-value-amber">{temp_val:.1f}°C</div>
            <div class="kpi-delta {temp_badge}">{temp_status}</div>
        </div>
        """)
    dap_val = int(latest_obs['dap'])
    if dap_val >= 40:
        harvest_badge = "kpi-delta-warn"
        harvest_text = "🚨 Batas Panen 40 HST!"
    elif dap_val >= 35:
        harvest_badge = "kpi-delta-up"
        harvest_text = f"🥬 Siap Panen (Sisa {40 - dap_val} Hari)"
    else:
        harvest_badge = "kpi-delta-info"
        harvest_text = f"🌿 Sisa {40 - dap_val} Hari ke Panen"

    with c4:
        st.html(f"""
        <div class="kpi-card kpi-emerald">
            <div class="kpi-label">🌱 Umur / Batas Panen</div>
            <div class="kpi-value kpi-value-green">{dap_val} / 40 HST</div>
            <div class="kpi-delta {harvest_badge}">{harvest_text}</div>
        </div>
        """)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.html(f"""
        <div class="kpi-card kpi-purple">
            <div class="kpi-label">🥬 Fase Pertumbuhan</div>
            <div class="kpi-value kpi-value-purple" style="font-size: 1.35rem; margin-top: 4px;">{latest_obs['criteria'].split(' (')[0]}</div>
            <div class="kpi-delta kpi-delta-purple">Kriteria Morfologi Asli</div>
        </div>
        """)
    with c6:
        st.html(f"""
        <div class="kpi-card kpi-green">
            <div class="kpi-label">🏷️ Total Batch Tanaman</div>
            <div class="kpi-value kpi-value-green">{total_plants} Unit</div>
            <div class="kpi-delta kpi-delta-up">{total_plants} Batch Terdaftar</div>
        </div>
        """)
    with c7:
        ano_class = "kpi-delta-warn" if active_anomalies > 0 else "kpi-delta-up"
        ano_text = f"{active_anomalies} Perhatian" if active_anomalies > 0 else "Semua Normal"
        st.html(f"""
        <div class="kpi-card kpi-amber">
            <div class="kpi-label">⚠️ Status Anomali</div>
            <div class="kpi-value kpi-value-amber">{active_anomalies} Anomali</div>
            <div class="kpi-delta {ano_class}">{ano_text}</div>
        </div>
        """)
    with c8:
        st.html(f"""
        <div class="kpi-card kpi-blue">
            <div class="kpi-label">📈 Data Observasi</div>
            <div class="kpi-value kpi-value-blue">{len(history_df)} Baris</div>
            <div class="kpi-delta kpi-delta-info">Frekuensi: 3x / Hari</div>
        </div>
        """)
        
    st.markdown("<br/>", unsafe_allow_html=True)

    # --- 2.5 PROGRESS MASA PANEN 40 HARI (MAXIMUM 40 DAYS HARVEST TRACKER) ---
    harvest_pct = min(round((dap_val / 40.0) * 100, 1), 100.0)
    bar_color = "#ef4444" if dap_val >= 40 else ("#f59e0b" if dap_val >= 35 else "linear-gradient(90deg, #10b981 0%, #059669 100%)")
    urgency_badge = (
        "<span style='background: #fee2e2; color: #b91c1c; font-weight: 800; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; border: 1px solid #fca5a5;'>🚨 BATAS PANEN 40 HST TERCAPAI (Wajib Panen Hari Ini)</span>"
        if dap_val >= 40 else
        (f"<span style='background: #fef3c7; color: #b45309; font-weight: 800; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; border: 1px solid #fde68a;'>🥬 JENDELA PANEN PRIMA (Sisa {40 - dap_val} Hari Menuju Batas 40 HST)</span>"
        if dap_val >= 35 else
        f"<span style='background: #ecfdf5; color: #047857; font-weight: 800; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; border: 1px solid #a7f3d0;'>🌱 SIKLUS PERTUMBUHAN AKTIF (Sisa {40 - dap_val} Hari ke Batas Panen 40 HST)</span>")
    )
    
    st.html(f"""
    <div style="background: #ffffff; border-radius: 14px; padding: 16px 20px; border: 1px solid #e2e8f0; margin-bottom: 20px; box-shadow: 0 4px 16px -2px rgba(0,0,0,0.04);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
            <div>
                <strong style="color: #064e3b; font-size: 1.02rem;">⏱️ Siklus Budidaya & Batas Panen Maksimal Tanaman Pakcoy</strong>
                <span style="color: #64748b; font-size: 0.85rem; margin-left: 8px;">(Standar Mutlak Budidaya: Maksimal 40 HST)</span>
            </div>
            {urgency_badge}
        </div>
        <div style="background: #e2e8f0; border-radius: 9999px; height: 10px; overflow: hidden; margin: 8px 0;">
            <div style="background: {bar_color}; width: {harvest_pct}%; height: 100%; border-radius: 9999px;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #64748b; font-weight: 600;">
            <span>0 HST (Semai)</span>
            <span>10 HST (Bibit)</span>
            <span>25 HST (Vegetatif)</span>
            <span>35 HST (Siap Panen)</span>
            <span style="color: #b91c1c; font-weight: 800;">40 HST (Batas Maksimal Panen)</span>
        </div>
    </div>
    """)
    
    # --- 3. KONDISI & STATUS TANAMAN ---
    st.markdown(f"<h3 style='color: #065f46;'>📌 Status Real-Time & Diagnosis: {selected_plant_id}</h3>", unsafe_allow_html=True)
    col_stat1, col_stat2 = st.columns([1.8, 1.2])
    
    with col_stat1:
        harvest_advice = "Tanaman telah mencapai batas maksimal panen 40 HST. Segera panen hari ini untuk mencegah serat mengeras dan risiko bolting." if dap_val >= 40 else (f"Tanaman berada dalam jendela panen prima dengan sisa waktu {40 - dap_val} hari sebelum batas akhir 40 HST." if dap_val >= 35 else f"Pertumbuhan vegetatif berlangsung optimal dengan estimasi panen {40 - dap_val} hari ke depan.")
        st.html(f"""
        <div class="plant-card" style="border-left: 5px solid #059669; background: #ffffff;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h4 style="margin: 0; color: #064e3b; font-size: 1.15rem;">🌿 Diagnostik Tanaman Terpilih</h4>
                <span class="status-badge {'status-optimal' if latest_obs['soil_condition']=='Optimal' and dap_val < 40 else 'status-warning'}">
                    ● Media: {latest_obs['soil_condition']}
                </span>
            </div>
            <p style="color: #334155; font-size: 0.92rem; margin-bottom: 8px;">
                <strong>Fokus Pengamatan:</strong> <span style="color: #047857; font-weight: 700;">{selected_plant_id}</span> | 
                <strong>Pencatatan Terakhir:</strong> Hari ke-{latest_obs['day']} (DAP {dap_val} HST / Maks 40 HST), Pukul {latest_obs['time']} WIB
            </p>
            <p style="color: #334155; font-size: 0.92rem; margin-bottom: 8px;">
                <strong>Fase & Morfologi:</strong> <em>{latest_obs['criteria']}</em> dengan indeks kematangan <strong>{latest_obs['maturity_level']}%</strong>.
            </p>
            <div style="background: {'#fef2f2' if dap_val >= 40 else '#f0fdf4'}; border: 1px solid {'#fecaca' if dap_val >= 40 else '#bbf7d0'}; border-radius: 10px; padding: 10px 14px; margin-top: 12px; font-size: 0.88rem; color: {'#991b1b' if dap_val >= 40 else '#166534'};">
                {'🚨' if dap_val >= 40 else '✅'} <strong>Status Panen (Maks 40 HST):</strong> {harvest_advice}
            </div>
        </div>
        """)
        
    with col_stat2:
        st.html("""
        <div class="plant-card" style="border-left: 5px solid #0284c7; background: #ffffff;">
            <h4 style="margin: 0 0 12px 0; color: #0369a1; font-size: 1.15rem;">💡 Indikator Budidaya Ideal</h4>
            <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.88rem;">
                <div style="display: flex; justify-content: space-between; padding: 6px 10px; background: #ecfdf5; border-radius: 8px;">
                    <span style="color: #047857; font-weight: 600;">⏱️ Masa Panen Maksimal</span>
                    <strong style="color: #059669;">Maks 40 HST</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 10px; background: #f0f9ff; border-radius: 8px;">
                    <span style="color: #0369a1; font-weight: 600;">💧 Kelembaban Optimal</span>
                    <strong style="color: #0284c7;">50% - 65%</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 10px; background: #fffbeb; border-radius: 8px;">
                    <span style="color: #b45309; font-weight: 600;">🌡️ Suhu Pertumbuhan</span>
                    <strong style="color: #d97706;">22°C - 28.5°C</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 10px; background: #f8fafc; border-radius: 8px;">
                    <span style="color: #475569; font-weight: 600;">🕒 Jadwal Pengamatan</span>
                    <strong style="color: #0f172a;">08:00, 12:00, 16:00 WIB</strong>
                </div>
            </div>
        </div>
        """)
        
    # --- 4. PANEL SENSOR DINAMIS (KELEMBABAN & SUHU REAL-TIME) ---
    st.markdown("<h3 style='color: #065f46; margin-top: 24px;'>📡 Panel Sensor Dinamis Mengikuti Simulasi Real-Time</h3>", unsafe_allow_html=True)
    tab_sens1, tab_sens2 = st.tabs(["💧 Panel Sensor Kelembaban Tanah", "🌡️ Panel Sensor Suhu Lingkungan"])
    with tab_sens1:
        render_soil_moisture_sensor_panel(history_df, sim_idx)
    with tab_sens2:
        render_temperature_sensor_panel(history_df, sim_idx)
        
    # --- 5. GRAFIK PERTUMBUHAN INTERAKTIF DENGAN WARNA HARMONIS ---
    st.markdown("<h3 style='color: #065f46; margin-top: 20px;'>📈 Grafik Pertumbuhan & Parameter Lingkungan</h3>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🌱 Kurva Kematangan (%)", "💧 Kelembaban Media (%)", "🌡️ Fluktuasi Suhu (°C)"])
    
    with tab1:
        fig_mat = go.Figure()
        fig_mat.add_trace(go.Scatter(
            x=history_df['day'],
            y=history_df['maturity_level'],
            mode="lines+markers",
            name="Kematangan Aktual (%)",
            line=dict(color="#059669", width=3, shape="spline"),
            marker=dict(size=6, color="#047857"),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.12)"
        ))
        # Marker on current simulation timestep
        fig_mat.add_trace(go.Scatter(
            x=[history_df['day'].iloc[sim_idx]],
            y=[history_df['maturity_level'].iloc[sim_idx]],
            mode="markers",
            name="Posisi Simulasi Aktif",
            marker=dict(size=14, color="#047857", symbol="star", line=dict(color="#ffffff", width=2))
        ))
        fig_mat.update_layout(
            title=f"Perkembangan Tingkat Kematangan: {selected_plant_id} (DAP {history_df['dap'].min()} s.d. {history_df['dap'].max()})",
            xaxis_title="Hari Pengamatan (Day)",
            yaxis_title="Kematangan (%)",
            hovermode="x unified",
            template="plotly_white",
            plot_bgcolor="rgba(248, 250, 252, 0.8)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=380
        )
        st.plotly_chart(fig_mat, use_container_width=True)
        
    with tab2:
        fig_moist = go.Figure()
        fig_moist.add_trace(go.Scatter(
            x=history_df['day'],
            y=history_df['soil_moisture'],
            mode="lines+markers",
            name="Kelembaban Media (%)",
            line=dict(color="#0284c7", width=3, shape="spline"),
            marker=dict(size=6, color="#0369a1"),
            fill="tozeroy",
            fillcolor="rgba(2, 132, 199, 0.10)"
        ))
        fig_moist.add_trace(go.Scatter(
            x=[history_df['day'].iloc[sim_idx]],
            y=[history_df['soil_moisture'].iloc[sim_idx]],
            mode="markers",
            name="Posisi Simulasi Aktif",
            marker=dict(size=14, color="#0284c7", symbol="circle", line=dict(color="#ffffff", width=2))
        ))
        fig_moist.add_hrect(
            y0=50, y1=65,
            line_width=0,
            fillcolor="#10b981",
            opacity=0.15,
            annotation_text="Rentang Optimal (50-65%)",
            annotation_position="top left"
        )
        fig_moist.update_layout(
            title=f"Dinamika Kelembaban Media Tanam: {selected_plant_id}",
            xaxis_title="Hari Pengamatan (Day)",
            yaxis_title="Kelembaban (%)",
            hovermode="x unified",
            template="plotly_white",
            plot_bgcolor="rgba(248, 250, 252, 0.8)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=380
        )
        st.plotly_chart(fig_moist, use_container_width=True)
        
    with tab3:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=history_df['day'],
            y=history_df['temperature'],
            mode="lines+markers",
            name="Suhu Mikro (°C)",
            line=dict(color="#f59e0b", width=3, shape="spline"),
            marker=dict(size=6, color="#d97706"),
            fill="tozeroy",
            fillcolor="rgba(245, 158, 11, 0.10)"
        ))
        fig_temp.add_trace(go.Scatter(
            x=[history_df['day'].iloc[sim_idx]],
            y=[history_df['temperature'].iloc[sim_idx]],
            mode="markers",
            name="Posisi Simulasi Aktif",
            marker=dict(size=14, color="#d97706", symbol="diamond", line=dict(color="#ffffff", width=2))
        ))
        fig_temp.add_hrect(
            y0=22, y1=28.5,
            line_width=0,
            fillcolor="#10b981",
            opacity=0.15,
            annotation_text="Rentang Optimal (22-28.5°C)",
            annotation_position="top left"
        )
        fig_temp.update_layout(
            title=f"Suhu Lingkungan Mikro: {selected_plant_id}",
            xaxis_title="Hari Pengamatan (Day)",
            yaxis_title="Suhu (°C)",
            hovermode="x unified",
            template="plotly_white",
            plot_bgcolor="rgba(248, 250, 252, 0.8)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=380
        )
        st.plotly_chart(fig_temp, use_container_width=True)
        
    # --- 6. PREDIKSI 4 HARI RINGKAS ---
    st.markdown("<h3 style='color: #065f46; margin-top: 20px;'>🔮 Ringkasan Proyeksi 4 Hari ke Depan (LSTM)</h3>", unsafe_allow_html=True)
    forecast_result = predict_growth_4days(selected_plant_id)
    if forecast_result:
        daily_sum = forecast_result["daily_summary_df"]
        st.dataframe(daily_sum, use_container_width=True, hide_index=True)
        st.info("💡 Buka menu **📈 Prediksi 4 Hari (LSTM)** pada navigasi sidebar untuk melihat tabel rinci 12 timestep dan simulasi per jam.")
    else:
        st.info("Data observasi tanaman ini memerlukan minimal 12 rekaman data untuk menghasilkan peramalan LSTM 4 hari.")

    # --- ADVANCE SIMULATION PLAYBACK LOOP IF PLAYING ---
    handle_simulation_playback(history_df)
