import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

def init_simulation_state(history_df, plant_id: str):
    """Initialize or sync the simulation playback state in session_state."""
    if "sim_plant_id" not in st.session_state or st.session_state["sim_plant_id"] != plant_id:
        st.session_state["sim_plant_id"] = plant_id
        st.session_state["sim_index"] = len(history_df) - 1
        st.session_state["sim_playing"] = False
        
    if "sim_index" not in st.session_state or st.session_state["sim_index"] >= len(history_df):
        st.session_state["sim_index"] = len(history_df) - 1
        
    if "sim_playing" not in st.session_state:
        st.session_state["sim_playing"] = False
        
    if "sim_speed" not in st.session_state:
        st.session_state["sim_speed"] = 1.0


def render_simulation_controller(history_df, plant_id: str, key_prefix: str = "ctrl"):
    """
    Renders time-travel simulation controls (Play/Pause, speed selector,
    stepper, and slider) that advance through observations (08:00 -> 12:00 -> 16:00 -> next day).
    """
    init_simulation_state(history_df, plant_id)
    total_steps = len(history_df)
    cur_idx = st.session_state["sim_index"]
    is_playing = st.session_state["sim_playing"]
    
    cur_obs = history_df.iloc[cur_idx]
    cur_day = int(cur_obs['day'])
    cur_dap = int(cur_obs['dap'])
    cur_time = cur_obs['time']
    
    # Status badges
    play_badge = (
        "<span style='background: #ecfdf5; color: #047857; font-weight: 800; padding: 4px 12px; border-radius: 9999px; border: 1.5px solid #10b981; font-size: 0.8rem; display: inline-flex; align-items: center; gap: 6px;'>"
        "<span class='pulse-dot'></span> ● SIMULASI AKTIF BERJALAN"
        "</span>"
        if is_playing else
        "<span style='background: #f1f5f9; color: #475569; font-weight: 700; padding: 4px 12px; border-radius: 9999px; border: 1px solid #cbd5e1; font-size: 0.8rem;'>"
        "⏸️ SIMULASI DIJEDA"
        "</span>"
    )
    
    st.html(f"""
    <div style="background: #ffffff; border: 1.5px solid #a7f3d0; border-radius: 16px; padding: 18px 22px; margin-bottom: 22px; box-shadow: 0 4px 20px -3px rgba(5, 150, 105, 0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">
            <div>
                <span style="font-size: 0.75rem; font-weight: 800; color: #047857; text-transform: uppercase; letter-spacing: 0.05em;">⏱️ Mesin Simulasi Deret Waktu</span>
                <h3 style="margin: 2px 0 0 0; color: #064e3b; font-size: 1.25rem;">
                    Hari ke-{cur_day} • DAP {cur_dap} HST • Pukul {cur_time} WIB
                </h3>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 0.8rem; color: #64748b; font-weight: 600;">Langkah: {cur_idx + 1} / {total_steps}</span>
                {play_badge}
            </div>
        </div>
    </div>
    """)
    
    # Controller Buttons Row
    c_btn_start, c_btn_prev, c_btn_play, c_btn_next, c_btn_end, c_speed = st.columns([1, 1, 1.4, 1, 1, 1.8])
    
    with c_btn_start:
        if st.button("⏮️ Awal", key=f"{key_prefix}_start", use_container_width=True, help="Kembali ke pembacaan pertama (Hari ke-1, 08:00)"):
            st.session_state["sim_index"] = 0
            st.session_state["sim_playing"] = False
            st.rerun()
            
    with c_btn_prev:
        if st.button("◀️ -1 Sesi", key=f"{key_prefix}_prev", use_container_width=True, help="Mundur 1 sesi pembacaan"):
            st.session_state["sim_index"] = max(0, cur_idx - 1)
            st.session_state["sim_playing"] = False
            st.rerun()
            
    with c_btn_play:
        if is_playing:
            if st.button("⏸️ Jeda Simulasi", key=f"{key_prefix}_pause", type="primary", use_container_width=True):
                st.session_state["sim_playing"] = False
                st.rerun()
        else:
            if st.button("▶️ Putar Simulasi", key=f"{key_prefix}_play", type="primary", use_container_width=True):
                st.session_state["sim_playing"] = True
                st.rerun()
                
    with c_btn_next:
        if st.button("▶️ +1 Sesi", key=f"{key_prefix}_next", use_container_width=True, help="Maju 1 sesi pembacaan (08:00 → 12:00 → 16:00 → Hari Berikutnya)"):
            st.session_state["sim_index"] = min(total_steps - 1, cur_idx + 1)
            st.session_state["sim_playing"] = False
            st.rerun()
            
    with c_btn_end:
        if st.button("⏭️ Terkini", key=f"{key_prefix}_end", use_container_width=True, help="Loncat ke observasi terbaru"):
            st.session_state["sim_index"] = total_steps - 1
            st.session_state["sim_playing"] = False
            st.rerun()
            
    with c_speed:
        speed_options = {
            "0.5x (Lambat 1.5s)": 1.5,
            "1.0x (Normal 0.8s)": 0.8,
            "2.0x (Cepat 0.4s)": 0.4,
            "4.0x (Kilat 0.2s)": 0.2
        }
        # Find current index
        cur_speed_val = st.session_state.get("sim_speed", 0.8)
        sel_speed_idx = 1
        for idx, (label, val) in enumerate(speed_options.items()):
            if val == cur_speed_val:
                sel_speed_idx = idx
                break
                
        chosen_label = st.selectbox(
            "Kecepatan Putar",
            list(speed_options.keys()),
            index=sel_speed_idx,
            key=f"{key_prefix}_speed_sel",
            label_visibility="collapsed"
        )
        st.session_state["sim_speed"] = speed_options[chosen_label]

    # Time-scrubber slider
    time_labels = [
        f"Hari {int(r['day'])} (DAP {int(r['dap'])}) - {r['time']}"
        for _, r in history_df.iterrows()
    ]
    
    new_idx = st.slider(
        "🎚️ Geser Garis Waktu Pengamatan (08:00 → 12:00 → 16:00):",
        min_value=0,
        max_value=total_steps - 1,
        value=cur_idx,
        key=f"{key_prefix}_slider",
        help="Geser slider untuk melompat langsung ke sesi waktu tertentu."
    )
    
    if new_idx != cur_idx and not is_playing:
        st.session_state["sim_index"] = new_idx
        st.rerun()
        
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    return cur_obs, cur_idx


def render_soil_moisture_sensor_panel(history_df, sim_index: int):
    """
    Renders the dedicated PANEL SENSOR KELEMBAPAN TANAH:
    - Gauge changing color according to Soil Condition:
      * Dry = Kuning/Oranye (#f59e0b)
      * Optimal = Hijau Emerald (#10b981)
      * Wet = Biru (#0284c7)
    - Current, Mean, Min, Max statistics
    - Hourly average breakdown (08:00, 12:00, 16:00)
    - Dynamic trend line chart following the simulation
    """
    cur_obs = history_df.iloc[sim_index]
    cur_moist = float(cur_obs['soil_moisture'])
    cur_cond = str(cur_obs['soil_condition']).strip()
    
    # Sub-dataframe up to simulation index
    sim_df = history_df.iloc[:sim_index + 1]
    
    # Statistical calculations
    mean_moist = float(sim_df['soil_moisture'].mean())
    min_moist = float(sim_df['soil_moisture'].min())
    max_moist = float(sim_df['soil_moisture'].max())
    
    # Hourly statistics across history
    h_stats = history_df.groupby('time')['soil_moisture'].mean().to_dict()
    avg_08 = h_stats.get('08:00', 66.3)
    avg_12 = h_stats.get('12:00', 62.6)
    avg_16 = h_stats.get('16:00', 60.9)
    
    # Color definition based on user prompt:
    # Dry = kuning/oranye, Optimal = hijau, Wet = biru
    if cur_cond.lower() == "dry" or cur_moist < 50.0:
        cond_theme = {
            "color": "#d97706",
            "bg": "#fffbeb",
            "border": "#fde68a",
            "label": "DRY (Kering - Perlu Irigasi)",
            "bar_color": "#f59e0b",
            "delta_badge": "kpi-delta-warn",
            "glow": "rgba(245, 158, 11, 0.25)"
        }
    elif cur_cond.lower() == "wet" or cur_moist > 75.0:
        cond_theme = {
            "color": "#0284c7",
            "bg": "#f0f9ff",
            "border": "#bae6fd",
            "label": "WET (Basah Jenuh - Kurangi Air)",
            "bar_color": "#0ea5e9",
            "delta_badge": "kpi-delta-info",
            "glow": "rgba(2, 132, 199, 0.25)"
        }
    else:
        cond_theme = {
            "color": "#059669",
            "bg": "#ecfdf5",
            "border": "#a7f3d0",
            "label": "OPTIMAL (Ideal 50% - 70%)",
            "bar_color": "#10b981",
            "delta_badge": "kpi-delta-up",
            "glow": "rgba(16, 185, 129, 0.25)"
        }

    st.html(f"""
    <div style="background: #ffffff; border: 1.5px solid {cond_theme['border']}; border-radius: 16px; padding: 20px 24px; margin-bottom: 20px; box-shadow: 0 4px 18px {cond_theme['glow']};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
            <div>
                <span style="font-size: 0.74rem; font-weight: 800; color: {cond_theme['color']}; text-transform: uppercase; letter-spacing: 0.05em;">
                    💧 Sensor Media Tanam Real-Time
                </span>
                <h3 style="margin: 2px 0 0 0; color: #064e3b; font-size: 1.3rem;">
                    PANEL SENSOR KELEMBAPAN TANAH
                </h3>
            </div>
            <span style="background: {cond_theme['bg']}; color: {cond_theme['color']}; border: 1.5px solid {cond_theme['border']}; font-weight: 800; font-size: 0.82rem; padding: 5px 14px; border-radius: 9999px;">
                Status: {cond_theme['label']}
            </span>
        </div>
    </div>
    """)

    # Top Grid: Gauge (Left) and 4 Quick Metrics + Hourly Patterns (Right)
    col_gauge, col_stats = st.columns([1.15, 1.35], gap="medium")
    
    with col_gauge:
        # Plotly Gauge Chart with Dynamic Color
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cur_moist,
            number={'suffix': "%", 'font': {'size': 38, 'color': cond_theme['color'], 'family': 'Outfit'}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Tingkat Kelembaban Terkini", 'font': {'size': 15, 'color': '#065f46'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1.5, 'tickcolor': "#64748b"},
                'bar': {'color': cond_theme['bar_color'], 'thickness': 0.3},
                'bgcolor': "#ffffff",
                'borderwidth': 1.5,
                'bordercolor': "#cbd5e1",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(245, 158, 11, 0.15)'},  # Dry
                    {'range': [50, 75], 'color': 'rgba(16, 185, 129, 0.22)'}, # Optimal
                    {'range': [75, 100], 'color': 'rgba(2, 132, 199, 0.18)'}  # Wet
                ],
                'threshold': {
                    'line': {'color': cond_theme['color'], 'width': 4},
                    'thickness': 0.8,
                    'value': cur_moist
                }
            }
        ))
        fig_g.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_g, use_container_width=True)
        
    with col_stats:
        # 4 Summary Metrics (Terkini, Rata-rata, Min, Max)
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        with c_m1:
            st.html(f"""
            <div class="kpi-card kpi-blue" style="padding: 12px; text-align: center;">
                <div class="kpi-label" style="font-size: 0.72rem;">💧 Terkini</div>
                <div class="kpi-value" style="font-size: 1.35rem; color: {cond_theme['color']};">{cur_moist:.1f}%</div>
                <div class="kpi-delta {cond_theme['delta_badge']}" style="font-size: 0.68rem;">{cur_cond}</div>
            </div>
            """)
        with c_m2:
            st.html(f"""
            <div class="kpi-card kpi-green" style="padding: 12px; text-align: center;">
                <div class="kpi-label" style="font-size: 0.72rem;">📊 Rata-rata</div>
                <div class="kpi-value kpi-value-green" style="font-size: 1.35rem;">{mean_moist:.1f}%</div>
                <div class="kpi-delta kpi-delta-up" style="font-size: 0.68rem;">Kumulatif</div>
            </div>
            """)
        with c_m3:
            st.html(f"""
            <div class="kpi-card kpi-amber" style="padding: 12px; text-align: center;">
                <div class="kpi-label" style="font-size: 0.72rem;">📉 Minimum</div>
                <div class="kpi-value kpi-value-amber" style="font-size: 1.35rem;">{min_moist:.1f}%</div>
                <div class="kpi-delta kpi-delta-warn" style="font-size: 0.68rem;">Titik Terkering</div>
            </div>
            """)
        with c_m4:
            st.html(f"""
            <div class="kpi-card kpi-purple" style="padding: 12px; text-align: center;">
                <div class="kpi-label" style="font-size: 0.72rem;">📈 Maksimum</div>
                <div class="kpi-value kpi-value-purple" style="font-size: 1.35rem;">{max_moist:.1f}%</div>
                <div class="kpi-delta kpi-delta-purple" style="font-size: 0.68rem;">Puncak Basah</div>
            </div>
            """)
            
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        
        # 3 Hourly Reading Averages (08:00, 12:00, 16:00)
        st.html("<p style='font-size: 0.8rem; font-weight: 800; color: #065f46; margin: 6px 0 6px 2px; text-transform: uppercase;'>🕒 Rata-rata Kelembaban Berdasarkan Jam Pengamatan:</p>")
        c_h1, c_h2, c_h3 = st.columns(3)
        with c_h1:
            st.html(f"""
            <div style="background: #f0fdf4; border: 1.5px solid #bbf7d0; border-radius: 10px; padding: 10px; text-align: center;">
                <div style="font-size: 0.72rem; color: #166534; font-weight: 700;">🌅 Pukul 08:00 WIB</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #059669; margin: 2px 0;">{avg_08:.1f}%</div>
                <div style="font-size: 0.68rem; color: #64748b;">Kondisi Basah Pagi</div>
            </div>
            """)
        with c_h2:
            st.html(f"""
            <div style="background: #f0f9ff; border: 1.5px solid #bae6fd; border-radius: 10px; padding: 10px; text-align: center;">
                <div style="font-size: 0.72rem; color: #0369a1; font-weight: 700;">☀️ Pukul 12:00 WIB</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #0284c7; margin: 2px 0;">{avg_12:.1f}%</div>
                <div style="font-size: 0.68rem; color: #64748b;">Evaporasi Siang</div>
            </div>
            """)
        with c_h3:
            st.html(f"""
            <div style="background: #fffbeb; border: 1.5px solid #fde68a; border-radius: 10px; padding: 10px; text-align: center;">
                <div style="font-size: 0.72rem; color: #92400e; font-weight: 700;">🌇 Pukul 16:00 WIB</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #d97706; margin: 2px 0;">{avg_16:.1f}%</div>
                <div style="font-size: 0.68rem; color: #64748b;">Stabilisasi Sore</div>
            </div>
            """)

    # Dynamic line chart following the simulation
    time_labels_all = [f"H{int(r['day'])} {r['time']}" for _, r in history_df.iterrows()]
    time_labels_sim = time_labels_all[:sim_index + 1]
    
    fig_line = go.Figure()
    
    # Faint full context
    fig_line.add_trace(go.Scatter(
        x=time_labels_all,
        y=history_df['soil_moisture'],
        mode="lines",
        name="Jalur Observasi Penuh",
        line=dict(color="#cbd5e1", width=1.5, dash="dot"),
        hoverinfo="skip"
    ))
    
    # Simulated active trace
    fig_line.add_trace(go.Scatter(
        x=time_labels_sim,
        y=sim_df['soil_moisture'],
        mode="lines+markers",
        name="Dinamika Simulasi",
        line=dict(color=cond_theme['bar_color'], width=3, shape="spline"),
        marker=dict(size=5, color=cond_theme['color']),
        fill="tozeroy",
        fillcolor=cond_theme['glow']
    ))
    
    # Current active simulated point
    fig_line.add_trace(go.Scatter(
        x=[time_labels_all[sim_index]],
        y=[cur_moist],
        mode="markers",
        name="Titik Waktu Aktif",
        marker=dict(size=14, color=cond_theme['color'], symbol="circle", line=dict(color="#ffffff", width=2.5))
    ))
    
    # Target range bands
    fig_line.add_hrect(
        y0=50, y1=75,
        fillcolor="#10b981", opacity=0.12,
        line_width=0,
        annotation_text="Rentang Optimal (50% - 75%)",
        annotation_position="top left"
    )
    
    fig_line.update_layout(
        title=f"Grafik Garis Tren Kelembaban Tanah Mengikuti Simulasi (Sesi {sim_index + 1} / {len(history_df)})",
        xaxis_title="Kronologi Hari & Jam Pembacaan",
        yaxis_title="Kelembaban Media (%)",
        hovermode="x unified",
        template="plotly_white",
        height=320,
        margin=dict(l=10, r=10, t=35, b=10),
        plot_bgcolor="rgba(248, 250, 252, 0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_line, use_container_width=True)


def render_temperature_sensor_panel(history_df, sim_index: int):
    """
    Renders the dedicated PANEL SENSOR SUHU:
    - Gauge and trend line for Temperature (°C) following the simulation
    - Current, Mean, Min, Max statistics
    - Daily diurnal pattern: Average Temperature at 08:00, 12:00, and 16:00
    - Dynamic trend line chart following the simulation
    """
    cur_obs = history_df.iloc[sim_index]
    cur_temp = float(cur_obs['temperature'])
    
    sim_df = history_df.iloc[:sim_index + 1]
    
    # Statistical calculations
    mean_temp = float(sim_df['temperature'].mean())
    min_temp = float(sim_df['temperature'].min())
    max_temp = float(sim_df['temperature'].max())
    
    # Hourly statistics across history (Diurnal Pattern)
    t_stats = history_df.groupby('time')['temperature'].mean().to_dict()
    avg_t_08 = t_stats.get('08:00', 25.4)
    avg_t_12 = t_stats.get('12:00', 29.9)
    avg_t_16 = t_stats.get('16:00', 28.0)
    
    # Temperature condition status
    if 22.0 <= cur_temp <= 28.5:
        temp_theme = {
            "color": "#059669",
            "bg": "#ecfdf5",
            "border": "#a7f3d0",
            "label": "OPTIMAL (22°C - 28.5°C)",
            "bar_color": "#10b981",
            "delta_badge": "kpi-delta-up",
            "glow": "rgba(16, 185, 129, 0.25)"
        }
    elif cur_temp < 22.0:
        temp_theme = {
            "color": "#0284c7",
            "bg": "#f0f9ff",
            "border": "#bae6fd",
            "label": "SUB-OPTIMAL (Sejuk)",
            "bar_color": "#38bdf8",
            "delta_badge": "kpi-delta-info",
            "glow": "rgba(2, 132, 199, 0.25)"
        }
    else:
        temp_theme = {
            "color": "#d97706",
            "bg": "#fffbeb",
            "border": "#fde68a",
            "label": "PERLU VENTILASI (>28.5°C)",
            "bar_color": "#f59e0b",
            "delta_badge": "kpi-delta-warn",
            "glow": "rgba(245, 158, 11, 0.25)"
        }

    st.html(f"""
    <div style="background: #ffffff; border: 1.5px solid {temp_theme['border']}; border-radius: 16px; padding: 20px 24px; margin-bottom: 20px; box-shadow: 0 4px 18px {temp_theme['glow']};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
            <div>
                <span style="font-size: 0.74rem; font-weight: 800; color: {temp_theme['color']}; text-transform: uppercase; letter-spacing: 0.05em;">
                    🌡️ Sensor Mikroklimat Real-Time
                </span>
                <h3 style="margin: 2px 0 0 0; color: #064e3b; font-size: 1.3rem;">
                    PANEL SENSOR SUHU LINGKUNGAN
                </h3>
            </div>
            <span style="background: {temp_theme['bg']}; color: {temp_theme['color']}; border: 1.5px solid {temp_theme['border']}; font-weight: 800; font-size: 0.82rem; padding: 5px 14px; border-radius: 9999px;">
                Status: {temp_theme['label']}
            </span>
        </div>
    </div>
    """)

    col_gauge, col_stats = st.columns([1.15, 1.35], gap="medium")
    
    with col_gauge:
        # Plotly Gauge for Temperature
        fig_tg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cur_temp,
            number={'suffix': "°C", 'font': {'size': 38, 'color': temp_theme['color'], 'family': 'Outfit'}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Suhu Lingkungan Terkini", 'font': {'size': 15, 'color': '#064e3b'}},
            gauge={
                'axis': {'range': [15, 40], 'tickwidth': 1.5, 'tickcolor': "#64748b"},
                'bar': {'color': temp_theme['bar_color'], 'thickness': 0.3},
                'bgcolor': "#ffffff",
                'borderwidth': 1.5,
                'bordercolor': "#cbd5e1",
                'steps': [
                    {'range': [15, 22], 'color': 'rgba(56, 189, 248, 0.15)'},  # Sejuk
                    {'range': [22, 28.5], 'color': 'rgba(16, 185, 129, 0.22)'},# Optimal
                    {'range': [28.5, 40], 'color': 'rgba(239, 68, 68, 0.18)'}  # Hangat / Panas
                ],
                'threshold': {
                    'line': {'color': temp_theme['color'], 'width': 4},
                    'thickness': 0.8,
                    'value': cur_temp
                }
            }
        ))
        fig_tg.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_tg, use_container_width=True)
        
    with col_stats:
        # 4 Summary Metrics (Terkini, Rata-rata, Min, Max)
        c_t1, c_t2, c_t3, c_t4 = st.columns(4)
        with c_t1:
            st.html(f"""
            <div class="kpi-card kpi-amber" style="padding: 12px; text-align: center;">
                <div class="kpi-label" style="font-size: 0.72rem;">🌡️ Terkini</div>
                <div class="kpi-value" style="font-size: 1.35rem; color: {temp_theme['color']};">{cur_temp:.1f}°C</div>
                <div class="kpi-delta {temp_theme['delta_badge']}" style="font-size: 0.68rem;">{temp_theme['label'].split(' ')[0]}</div>
            </div>
            """)
        with c_t2:
            st.html(f"""
            <div class="kpi-card kpi-green" style="padding: 12px; text-align: center;">
                <div class="kpi-label" style="font-size: 0.72rem;">📊 Rata-rata</div>
                <div class="kpi-value kpi-value-green" style="font-size: 1.35rem;">{mean_temp:.1f}°C</div>
                <div class="kpi-delta kpi-delta-up" style="font-size: 0.68rem;">Kumulatif</div>
            </div>
            """)
        with c_t3:
            st.html(f"""
            <div class="kpi-card kpi-blue" style="padding: 12px; text-align: center;">
                <div class="kpi-label" style="font-size: 0.72rem;">❄️ Minimum</div>
                <div class="kpi-value kpi-value-blue" style="font-size: 1.35rem;">{min_temp:.1f}°C</div>
                <div class="kpi-delta kpi-delta-info" style="font-size: 0.68rem;">Paling Sejuk</div>
            </div>
            """)
        with c_t4:
            st.html(f"""
            <div class="kpi-card kpi-purple" style="padding: 12px; text-align: center;">
                <div class="kpi-label" style="font-size: 0.72rem;">🔥 Maksimum</div>
                <div class="kpi-value kpi-value-purple" style="font-size: 1.35rem;">{max_temp:.1f}°C</div>
                <div class="kpi-delta kpi-delta-warn" style="font-size: 0.68rem;">Puncak Panas</div>
            </div>
            """)
            
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        
        # 3 Hourly Diurnal Pattern Cards (08:00, 12:00, 16:00)
        st.html("<p style='font-size: 0.8rem; font-weight: 800; color: #065f46; margin: 6px 0 6px 2px; text-transform: uppercase;'>☀️ Pola Harian Suhu Lingkungan (Diurnal Temperature Cycle):</p>")
        c_th1, c_th2, c_th3 = st.columns(3)
        with c_th1:
            st.html(f"""
            <div style="background: #f0fdf4; border: 1.5px solid #bbf7d0; border-radius: 10px; padding: 10px; text-align: center;">
                <div style="font-size: 0.72rem; color: #166534; font-weight: 700;">🌅 Pukul 08:00 WIB</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #059669; margin: 2px 0;">{avg_t_08:.1f}°C</div>
                <div style="font-size: 0.68rem; color: #64748b;">Sejuk Pagi Hari</div>
            </div>
            """)
        with c_th2:
            st.html(f"""
            <div style="background: #fffbeb; border: 1.5px solid #fde68a; border-radius: 10px; padding: 10px; text-align: center;">
                <div style="font-size: 0.72rem; color: #b45309; font-weight: 700;">☀️ Pukul 12:00 WIB</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #d97706; margin: 2px 0;">{avg_t_12:.1f}°C</div>
                <div style="font-size: 0.68rem; color: #b45309; font-weight: 700;">Puncak Terik Siang</div>
            </div>
            """)
        with c_th3:
            st.html(f"""
            <div style="background: #eff6ff; border: 1.5px solid #bfdbfe; border-radius: 10px; padding: 10px; text-align: center;">
                <div style="font-size: 0.72rem; color: #1e40af; font-weight: 700;">🌇 Pukul 16:00 WIB</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #2563eb; margin: 2px 0;">{avg_t_16:.1f}°C</div>
                <div style="font-size: 0.68rem; color: #64748b;">Kondisi Mendingin</div>
            </div>
            """)

    # Dynamic line chart following the simulation
    time_labels_all = [f"H{int(r['day'])} {r['time']}" for _, r in history_df.iterrows()]
    time_labels_sim = time_labels_all[:sim_index + 1]
    
    fig_tline = go.Figure()
    
    # Faint full context
    fig_tline.add_trace(go.Scatter(
        x=time_labels_all,
        y=history_df['temperature'],
        mode="lines",
        name="Jalur Observasi Penuh",
        line=dict(color="#cbd5e1", width=1.5, dash="dot"),
        hoverinfo="skip"
    ))
    
    # Simulated active trace
    fig_tline.add_trace(go.Scatter(
        x=time_labels_sim,
        y=sim_df['temperature'],
        mode="lines+markers",
        name="Fluktuasi Suhu Simulasi",
        line=dict(color=temp_theme['bar_color'], width=3, shape="spline"),
        marker=dict(size=5, color=temp_theme['color']),
        fill="tozeroy",
        fillcolor=temp_theme['glow']
    ))
    
    # Current active simulated point
    fig_tline.add_trace(go.Scatter(
        x=[time_labels_all[sim_index]],
        y=[cur_temp],
        mode="markers",
        name="Titik Waktu Aktif",
        marker=dict(size=14, color=temp_theme['color'], symbol="star", line=dict(color="#ffffff", width=2.5))
    ))
    
    # Target range bands
    fig_tline.add_hrect(
        y0=22.0, y1=28.5,
        fillcolor="#10b981", opacity=0.12,
        line_width=0,
        annotation_text="Rentang Optimal (22°C - 28.5°C)",
        annotation_position="top left"
    )
    
    fig_tline.update_layout(
        title=f"Grafik Tren Fluktuasi Suhu Mengikuti Simulasi (Sesi {sim_index + 1} / {len(history_df)})",
        xaxis_title="Kronologi Hari & Jam Pembacaan",
        yaxis_title="Suhu (°C)",
        hovermode="x unified",
        template="plotly_white",
        height=320,
        margin=dict(l=10, r=10, t=35, b=10),
        plot_bgcolor="rgba(248, 250, 252, 0.8)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_tline, use_container_width=True)


def handle_simulation_playback(history_df):
    """
    Called at the very end of the page rendering to advance the simulation
    to the next observation reading when 'sim_playing' is True.
    """
    if st.session_state.get("sim_playing", False):
        total_steps = len(history_df)
        speed = st.session_state.get("sim_speed", 0.8)
        time.sleep(speed)
        cur_idx = st.session_state.get("sim_index", total_steps - 1)
        next_idx = (cur_idx + 1) % total_steps
        st.session_state["sim_index"] = next_idx
        st.rerun()
