import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from services.data_service import (
    get_all_plants, get_latest_observation, get_growth_history, get_alerts
)
from services.anomaly_service import detect_anomalies_for_plant
from web.sensor_panels import (
    render_simulation_controller,
    render_soil_moisture_sensor_panel,
    render_temperature_sensor_panel,
    handle_simulation_playback
)

def render_dashboard(selected_plant_id: str):
    plants_df = get_all_plants()
    history_df = get_growth_history(selected_plant_id, limit=5000)
    
    if len(history_df) == 0:
        st.warning("Data observasi belum tersedia untuk tanaman ini.")
        return

    # --- 1. HERO BANNER ---
    st.html(f"""
    <div class="hero-container">
        <div class="hero-badge">🌱 Smart Pakcoy Monitoring & Irrigation IoT</div>
        <h1 class="hero-title">Dashboard Utama Pertumbuhan Pakcoy</h1>
        <p class="hero-subtitle">
            Pemantauan presisi budidaya pakcoy 40 hari terintegrasi kontrol pompa irigasi otomatis real-time 
            dan telemetri sensor kelembaban tanah kontinu.
        </p>
    </div>
    """)
    
    # --- 1.5 SIMULASI TELEMETRI (DEFAULT: JALAN TERUS) ---
    if "sim_playing" not in st.session_state:
        st.session_state["sim_playing"] = True  # Berjalan terus by default
    
    sim_obs, sim_idx = render_simulation_controller(history_df, selected_plant_id, key_prefix="dash")
    latest_obs = sim_obs
    dap_val = int(latest_obs.get('dap', latest_obs.get('day', 1)))
    cur_moist = float(latest_obs.get('soil_moisture', 55.0))
    cur_temp = float(latest_obs.get('temperature', 26.0))
    leaf_w = float(latest_obs.get('leaf_width', 5.0))
    
    # --- LOGIKA STATUS POMPA REAL-TIME ---
    pump_db_status = latest_obs.get('pump_status', 'OFF')
    if "manual_pump_override" in st.session_state:
        pump_active = st.session_state["manual_pump_override"]
    else:
        pump_active = (cur_moist < 50.0) or (pump_db_status == 'ON')
        
    pump_status_text = "AKTIF (MENYIRAM 💧)" if pump_active else "STANDBY (MATI ⏸️)"
    pump_card_bg = "#ecfdf5" if pump_active else "#f8fafc"
    pump_border = "#10b981" if pump_active else "#cbd5e1"
    pump_title_color = "#047857" if pump_active else "#475569"
    pump_badge_bg = "#d1fae5" if pump_active else "#e2e8f0"
    pump_badge_color = "#065f46" if pump_active else "#334155"

    # --- 2. PROGRESS MASA PANEN 40 HARI ---
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
    
    # --- 3. STATUS REAL-TIME & DIAGNOSIS (DIPERTAHANKAN SESUAI PERMINTAAN PENGGUNA) ---
    st.markdown(f"<h3 style='color: #065f46;'>📌 Status Real-Time & Diagnosis: {selected_plant_id}</h3>", unsafe_allow_html=True)
    col_stat1, col_stat2 = st.columns([1.8, 1.2])
    
    with col_stat1:
        harvest_advice = "Tanaman telah mencapai batas maksimal panen 40 HST. Segera panen hari ini untuk mencegah serat mengeras dan risiko bolting." if dap_val >= 40 else (f"Tanaman berada dalam jendela panen prima dengan sisa waktu {40 - dap_val} hari sebelum batas akhir 40 HST." if dap_val >= 35 else f"Pertumbuhan vegetatif berlangsung optimal dengan estimasi panen {40 - dap_val} hari ke depan.")
        st.html(f"""
        <div class="plant-card" style="border-left: 5px solid #059669; background: #ffffff; border-radius: 12px; padding: 18px 20px; border: 1px solid #e2e8f0; box-shadow: 0 4px 16px rgba(0,0,0,0.03);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h4 style="margin: 0; color: #064e3b; font-size: 1.15rem; display: flex; align-items: center; gap: 8px;">
                    <span>🌿</span> Diagnostik Tanaman Terpilih
                </h4>
                <span class="status-badge" style="background: #dcfce7; color: #166534; font-size: 0.78rem; font-weight: 700; padding: 4px 12px; border-radius: 9999px;">
                    ● Media: {latest_obs.get('soil_condition', 'Optimal')}
                </span>
            </div>
            <p style="color: #334155; font-size: 0.92rem; margin-bottom: 8px; line-height: 1.5;">
                <strong>Fokus Pengamatan:</strong> <span style="color: #047857; font-weight: 700;">{selected_plant_id}</span> | 
                <strong>Pencatatan Terakhir:</strong> Hari ke-{latest_obs.get('day', 1)} (DAP {dap_val} HST / Maks 40 HST), Pukul {latest_obs.get('time', '16:00')} WIB
            </p>
            <p style="color: #334155; font-size: 0.92rem; margin-bottom: 12px; line-height: 1.5;">
                <strong>Fase & Morfologi:</strong> <em>{latest_obs.get('criteria', 'Vegetatif')}</em> dengan indeks kematangan <strong>{latest_obs.get('maturity_level', 36.0):.1f}%</strong>.
            </p>
            <div style="background: {'#fef2f2' if dap_val >= 40 else '#f0fdf4'}; border: 1px solid {'#fecaca' if dap_val >= 40 else '#bbf7d0'}; border-radius: 10px; padding: 10px 14px; margin-top: 12px; font-size: 0.88rem; color: {'#991b1b' if dap_val >= 40 else '#166534'};">
                {'🚨' if dap_val >= 40 else '✅'} <strong>Status Panen (Maks 40 HST):</strong> {harvest_advice}
            </div>
        </div>
        """)
        
    with col_stat2:
        st.html("""
        <div class="plant-card" style="border-left: 5px solid #0284c7; background: #ffffff; border-radius: 12px; padding: 18px 20px; border: 1px solid #e2e8f0; box-shadow: 0 4px 16px rgba(0,0,0,0.03);">
            <h4 style="margin: 0 0 12px 0; color: #0369a1; font-size: 1.15rem; display: flex; align-items: center; gap: 8px;">
                <span>💡</span> Indikator Budidaya Ideal
            </h4>
            <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.88rem;">
                <div style="display: flex; justify-content: space-between; padding: 6px 10px; background: #ecfdf5; border-radius: 8px;">
                    <span style="color: #047857; font-weight: 600;">⏱️ Masa Panen Maksimal</span>
                    <strong style="color: #059669;">Maks 40 HST</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 10px; background: #f0f9ff; border-radius: 8px;">
                    <span style="color: #0369a1; font-weight: 600;">💧 Kelembaban Optimal</span>
                    <strong style="color: #0284c7;">50% – 65%</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 10px; background: #fffbeb; border-radius: 8px;">
                    <span style="color: #b45309; font-weight: 600;">🌡️ Suhu Pertumbuhan</span>
                    <strong style="color: #d97706;">22°C – 28.5°C</strong>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 10px; background: #f8fafc; border-radius: 8px;">
                    <span style="color: #475569; font-weight: 600;">🕒 Jadwal Pengamatan</span>
                    <strong style="color: #0f172a;">08:00, 12:00, 16:00 WIB</strong>
                </div>
            </div>
        </div>
        """)
        
    st.markdown("<br/>", unsafe_allow_html=True)

    # --- 4. STATUS POMPA IRIGASI REAL-TIME & KONTROL IOT ---
    st.markdown(f"<h3 style='color: #065f46;'>🚰 Status Pompa Irigasi Real-Time & Otomasi IoT</h3>", unsafe_allow_html=True)
    
    col_pump1, col_pump2 = st.columns([1.6, 1.2])
    
    with col_pump1:
        st.html(f"""
        <div style="background: {pump_card_bg}; border: 2px solid {pump_border}; border-radius: 14px; padding: 18px 22px; box-shadow: 0 4px 16px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="font-size: 1.8rem;">{'🚿' if pump_active else '🚰'}</div>
                    <div>
                        <div style="font-size: 0.75rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">STATUS TELEMETRI POMPA</div>
                        <h4 style="margin: 0; color: {pump_title_color}; font-size: 1.35rem; font-weight: 800;">{pump_status_text}</h4>
                    </div>
                </div>
                <span style="background: {pump_badge_bg}; color: {pump_badge_color}; font-size: 0.8rem; font-weight: 800; padding: 4px 12px; border-radius: 8px;">
                    {'RELAY ON (220V)' if pump_active else 'RELAY STANDBY'}
                </span>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 14px;">
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">KELEMBABAN</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: {'#0284c7' if cur_moist >= 50 else '#dc2626'}; margin-top: 2px;">{cur_moist:.1f}%</div>
                </div>
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">AMBANG NYALA</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #d97706; margin-top: 2px;">&lt; 50.0%</div>
                </div>
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; text-align: center;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">DEBIT IRIGASI</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #059669; margin-top: 2px;">{'1.2 L/menit' if pump_active else '0.0 L/menit'}</div>
                </div>
            </div>
            
            <p style="font-size: 0.84rem; color: #475569; margin: 12px 0 0 0; line-height: 1.4;">
                💡 <strong>Kaidah Otomasi:</strong> Sistem IoT menyalakan pompa air secara otomatis saat sensor mendeteksi kelembaban tanah di bawah 50.0%, dan otomatis beralih ke mode standby saat kelembaban optimal terjaga.
            </p>
        </div>
        """)
        
    with col_pump2:
        st.html("""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px 20px; box-shadow: 0 4px 16px rgba(0,0,0,0.03);">
            <h4 style="margin: 0 0 10px 0; color: #0f172a; font-size: 1.05rem;">⚙️ Kontrol Manual Pompa</h4>
            <p style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px;">
                Pengguna dapat melakukan uji penyiraman manual di luar jadwal otomatis IoT:
            </p>
        </div>
        """)
        c_pbtn1, c_pbtn2 = st.columns(2)
        with c_pbtn1:
            if st.button("💧 Nyalakan Pompa", use_container_width=True, type="primary" if not pump_active else "secondary"):
                st.session_state["manual_pump_override"] = True
                st.rerun()
        with c_pbtn2:
            if st.button("⏹️ Matikan Pompa", use_container_width=True, type="secondary"):
                st.session_state["manual_pump_override"] = False
                st.rerun()
                
        if st.button("🔄 Kembalikan ke Mode Otomatis IoT", use_container_width=True):
            st.session_state.pop("manual_pump_override", None)
            st.rerun()

    # --- 5. GRAFIK GARIS TREN KELEMBABAN JALAN TERUS (CONTINUOUS MOISTURE STREAMING) ---
    st.markdown("<h3 style='color: #065f46; margin-top: 26px;'>🌊 Grafik Garis Tren Kelembaban (Live Telemetri Berjalan Terus)</h3>", unsafe_allow_html=True)
    
    # Ambil sliding window hingga posisi simulasi aktif
    window_size = 50
    start_w = max(0, sim_idx - window_size + 1)
    stream_df = history_df.iloc[start_w:sim_idx + 1].copy()
    
    # Format sumbu X dengan Hari & Jam
    stream_x = [f"H{r['day']} {r['time']}" for _, r in stream_df.iterrows()]
    
    fig_moist_live = go.Figure()
    
    # Garis tren kelembaban
    fig_moist_live.add_trace(go.Scatter(
        x=stream_x,
        y=stream_df['soil_moisture'],
        mode="lines+markers",
        name="Kelembaban Tanah (%)",
        line=dict(color="#0284c7", width=3, shape="spline"),
        marker=dict(size=6, color="#0369a1"),
        fill="tozeroy",
        fillcolor="rgba(2, 132, 199, 0.12)"
    ))
    
    # Titik Real-Time Berjalan
    fig_moist_live.add_trace(go.Scatter(
        x=[stream_x[-1]],
        y=[cur_moist],
        mode="markers",
        name="Posisi Sensor Real-Time",
        marker=dict(size=14, color="#047857" if cur_moist >= 50 else "#ef4444", symbol="circle", line=dict(color="#ffffff", width=2.5))
    ))
    
    # Ambang batas optimal kelembaban
    fig_moist_live.add_hrect(
        y0=50, y1=65,
        line_width=0,
        fillcolor="#10b981",
        opacity=0.15,
        annotation_text="Rentang Optimal (50% - 65%)",
        annotation_position="top left"
    )
    
    # Garis batas pemicu pompa (50%)
    fig_moist_live.add_hline(
        y=50.0,
        line_dash="dash",
        line_color="#ef4444",
        line_width=1.5,
        annotation_text="Batas Pemicu Pompa (< 50% ON)",
        annotation_position="bottom left"
    )
    
    fig_moist_live.update_layout(
        title=f"Dinamika Kelembaban Media Tanam Kontinu: {selected_plant_id} (Titik Observasi ke-{sim_idx + 1} / {len(history_df)})",
        xaxis_title="Kronologi Waktu Sensor (Hari & Jam)",
        yaxis_title="Kelembaban Media Tanam (%)",
        hovermode="x unified",
        template="plotly_white",
        plot_bgcolor="rgba(248, 250, 252, 0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_moist_live, use_container_width=True)
    
    st.html("""
    <div style="display: flex; justify-content: space-between; align-items: center; background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 8px 16px; margin-top: -6px; margin-bottom: 20px;">
        <span style="font-size: 0.82rem; color: #166534; font-weight: 700;">
            🟢 Mode Streaming Aktif: Grafik kelembaban terus berjalan memperbarui data observasi IoT secara otomatis.
        </span>
        <span style="font-size: 0.78rem; color: #15803d; font-weight: 800;">
            Siklus 40 Hari • 1.600+ Data Telemetri
        </span>
    </div>
    """)

    # --- 6. PANEL SENSOR DINAMIS (SUHU & KEMATANGAN) ---
    st.markdown("<h3 style='color: #065f46; margin-top: 10px;'>📡 Panel Sensor Dinamis & Fluktuasi Suhu</h3>", unsafe_allow_html=True)
    tab_sens1, tab_sens2 = st.tabs(["🌡️ Panel Sensor Suhu Lingkungan", "🌱 Grafik Perkembangan Kematangan (%)"])
    
    with tab_sens1:
        render_temperature_sensor_panel(history_df, sim_idx)
        
    with tab_sens2:
        fig_mat = go.Figure()
        fig_mat.add_trace(go.Scatter(
            x=history_df['day'],
            y=history_df['maturity_level'],
            mode="lines+markers",
            name="Kematangan Aktual (%)",
            line=dict(color="#059669", width=2.5, shape="spline"),
            marker=dict(size=4, color="#047857"),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.10)"
        ))
        fig_mat.update_layout(
            title=f"Kurva Perkembangan Kematangan: {selected_plant_id} (DAP {history_df['dap'].min()} s.d. {history_df['dap'].max()} HST)",
            xaxis_title="Hari Pengamatan (Day)",
            yaxis_title="Kematangan (%)",
            hovermode="x unified",
            template="plotly_white",
            plot_bgcolor="rgba(248, 250, 252, 0.8)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=340
        )
        st.plotly_chart(fig_mat, use_container_width=True)

    # --- ADVANCE SIMULATION PLAYBACK LOOP (AGAR GRAFIK JALAN TERUS) ---
    handle_simulation_playback(history_df)
