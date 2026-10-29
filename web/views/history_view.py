import streamlit as st
import pandas as pd
from services.data_service import get_all_plants, get_all_observations_df

def render_history():
    st.html("""
    <div class="hero-container" style="background: linear-gradient(135deg, #064e3b 0%, #0369a1 50%, #059669 100%);">
        <div class="hero-badge">📜 Database Telemetri & Log Observasi</div>
        <h1 class="hero-title">Histori Observasi Pertumbuhan Pakcoy</h1>
        <p class="hero-subtitle">
            Akses riwayat lengkap telemetri sensorik 40 hari siklus budidaya dengan minimal 1.600 rekaman data, 
            mencakup kelembaban media tanam, suhu lingkungan mikro, status aktivasi pompa air, dan pertambahan lebar daun.
        </p>
    </div>
    """)
    
    # Load all observations
    obs_df = get_all_observations_df()
    total_records = len(obs_df)
    
    if obs_df.empty:
        st.warning("Data histori observasi belum tersedia.")
        return

    # --- 1. RINGKASAN STATISTIK DATASET 1.600+ DATA ---
    st.markdown("<h3 style='color: #065f46; margin-bottom: 14px;'>📊 Ringkasan Dataset Telemetri 40 Hari</h3>", unsafe_allow_html=True)
    
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    
    with col_s1:
        st.html(f"""
        <div style="background: #ffffff; border: 1.5px solid #a7f3d0; border-radius: 12px; padding: 14px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="font-size: 0.72rem; font-weight: 800; color: #047857; text-transform: uppercase;">TOTAL REKAMAN DATA</div>
            <div style="font-size: 1.45rem; font-weight: 800; color: #065f46; margin-top: 4px;">{total_records:,}</div>
            <div style="font-size: 0.75rem; color: #059669; font-weight: 600; margin-top: 2px;">≥ 1.600 Baris Data</div>
        </div>
        """)
        
    with col_s2:
        max_day = int(obs_df['day'].max())
        st.html(f"""
        <div style="background: #ffffff; border: 1.5px solid #bfdbfe; border-radius: 12px; padding: 14px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="font-size: 0.72rem; font-weight: 800; color: #0284c7; text-transform: uppercase;">RENTANG BUDIDAYA</div>
            <div style="font-size: 1.45rem; font-weight: 800; color: #0369a1; margin-top: 4px;">{max_day} Hari</div>
            <div style="font-size: 0.75rem; color: #0284c7; font-weight: 600; margin-top: 2px;">Siklus Penuh 40 HST</div>
        </div>
        """)
        
    with col_s3:
        avg_moist = float(obs_df['soil_moisture'].mean())
        st.html(f"""
        <div style="background: #ffffff; border: 1.5px solid #bbf7d0; border-radius: 12px; padding: 14px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="font-size: 0.72rem; font-weight: 800; color: #059669; text-transform: uppercase;">RATA-RATA KELEMBABAN</div>
            <div style="font-size: 1.45rem; font-weight: 800; color: #047857; margin-top: 4px;">{avg_moist:.1f}%</div>
            <div style="font-size: 0.75rem; color: #166534; font-weight: 600; margin-top: 2px;">Target: 50% - 65%</div>
        </div>
        """)
        
    with col_s4:
        avg_leaf = float(obs_df['leaf_width'].mean()) if 'leaf_width' in obs_df.columns else 6.5
        st.html(f"""
        <div style="background: #ffffff; border: 1.5px solid #ddd6fe; border-radius: 12px; padding: 14px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="font-size: 0.72rem; font-weight: 800; color: #7c3aed; text-transform: uppercase;">RATA-RATA LEBAR DAUN</div>
            <div style="font-size: 1.45rem; font-weight: 800; color: #6d28d9; margin-top: 4px;">{avg_leaf:.1f} cm</div>
            <div style="font-size: 0.75rem; color: #7c3aed; font-weight: 600; margin-top: 2px;">Panen: ≥ 10.0 cm</div>
        </div>
        """)
        
    with col_s5:
        pump_on_count = int((obs_df['pump_status'] == 'ON').sum()) if 'pump_status' in obs_df.columns else 0
        st.html(f"""
        <div style="background: #ffffff; border: 1.5px solid #fed7aa; border-radius: 12px; padding: 14px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
            <div style="font-size: 0.72rem; font-weight: 800; color: #c2410c; text-transform: uppercase;">AKTIVASI POMPA AIR</div>
            <div style="font-size: 1.45rem; font-weight: 800; color: #ea580c; margin-top: 4px;">{pump_on_count:,}x</div>
            <div style="font-size: 0.75rem; color: #c2410c; font-weight: 600; margin-top: 2px;">Irigasi Otomatis IoT</div>
        </div>
        """)

    st.markdown("<br/>", unsafe_allow_html=True)
    
    # --- 2. FILTER DATA INTERAKTIF ---
    st.markdown("<h3 style='color: #065f46;'>🔍 Filter Log Pengamatan</h3>", unsafe_allow_html=True)
    
    c_f1, c_f2, c_f3 = st.columns(3)
    with c_f1:
        plant_options = ["Semua Tanaman"] + sorted(obs_df['plant_id'].unique().tolist())
        sel_plant = st.selectbox("Filter ID Tanaman / Batch:", plant_options)
    with c_f2:
        day_range = st.slider("Filter Rentang Hari Pengamatan (Day 1 - 40):", 1, 40, (1, 40))
    with c_f3:
        pump_options = ["Semua Status Pompa", "ON (Sedang Menyiram)", "OFF (Standby)"]
        sel_pump = st.selectbox("Filter Status Pompa:", pump_options)
        
    # Apply filtering
    filtered_df = obs_df.copy()
    if sel_plant != "Semua Tanaman":
        filtered_df = filtered_df[filtered_df['plant_id'] == sel_plant]
    filtered_df = filtered_df[(filtered_df['day'] >= day_range[0]) & (filtered_df['day'] <= day_range[1])]
    
    if sel_pump == "ON (Sedang Menyiram)":
        filtered_df = filtered_df[filtered_df['pump_status'] == 'ON']
    elif sel_pump == "OFF (Standby)":
        filtered_df = filtered_df[filtered_df['pump_status'] == 'OFF']

    # --- 3. DOWNLOAD CSV & TABLE DISPLAY ---
    c_head1, c_head2 = st.columns([2, 1])
    with c_head1:
        st.write(f"Menampilkan **{len(filtered_df):,}** baris data hasil penyaringan:")
    with c_head2:
        csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Dataset Terfilter (CSV)",
            data=csv_bytes,
            file_name=f"histori_pakcoy_{sel_plant.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Format table for elegant presentation
    display_cols = [
        "id", "plant_id", "day", "dap", "time", 
        "soil_moisture", "temperature", "pump_status", 
        "leaf_width", "maturity_level", "criteria"
    ]
    avail_cols = [c for c in display_cols if c in filtered_df.columns]
    
    renamed_map = {
        "id": "ID",
        "plant_id": "Batch Tanaman",
        "day": "Hari",
        "dap": "HST",
        "time": "Waktu",
        "soil_moisture": "Kelembaban (%)",
        "temperature": "Suhu (°C)",
        "pump_status": "Status Pompa",
        "leaf_width": "Lebar Daun (cm)",
        "maturity_level": "Kematangan (%)",
        "criteria": "Fase Kriteria"
    }
    
    st.dataframe(
        filtered_df[avail_cols].rename(columns=renamed_map),
        use_container_width=True,
        hide_index=True,
        height=520
    )
