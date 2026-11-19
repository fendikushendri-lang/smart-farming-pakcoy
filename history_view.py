import streamlit as st
import pandas as pd
import json
from services.data_service import get_all_plants, get_all_observations_df, get_activity_logs

@st.dialog("📋 Detail Aktivitas Sistem & AI")
def show_activity_detail(row_data):
    waktu = row_data['waktu']
    jenis = row_data['jenis']
    proses = row_data['proses']
    status = row_data['status']
    raw_detail = row_data.get('detail_json', '{}')
    
    try:
        detail_data = json.loads(raw_detail) if raw_detail else {}
    except Exception:
        detail_data = {}

    st.html(f"""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px 18px; margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 0.75rem; font-weight: 800; color: #64748b; text-transform: uppercase;">
                IDENTIFIKASI AUDIT AKTIVITAS
            </span>
            <span style="background: #dcfce7; color: #166534; font-size: 0.78rem; font-weight: 700; padding: 3px 12px; border-radius: 9999px; border: 1px solid #bbf7d0;">
                ✔ {status}
            </span>
        </div>
        <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-bottom: 4px;">
            {proses}
        </div>
        <div style="font-size: 0.85rem; color: #64748b;">
            🕒 Waktu Eksekusi: <strong>{waktu} WIB</strong> • Kategori: <strong>{jenis}</strong>
        </div>
    </div>
    """)
    
    if detail_data:
        st.markdown("<p style='font-weight: 700; font-size: 0.88rem; color: #1e293b; margin-bottom: 6px;'>Parameter & Output Proses:</p>", unsafe_allow_html=True)
        st.json(detail_data)
        
    if st.button("Tutup", use_container_width=True, type="secondary"):
        st.rerun()

def render_history():
    # Inject Custom CSS for pill styling & clean table appearance matching user screenshot
    st.html("""
    <style>
    /* Custom Styling for Activity Filter Pills matching screenshot */
    div[data-testid="stButtonGroup"] {
        gap: 8px !important;
        margin-top: 4px !important;
    }
    div[data-testid="stButtonGroup"] button {
        border-radius: 8px !important;
        padding: 7px 22px !important;
        font-size: 0.9rem !important;
        border: none !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stButtonGroup"] button[aria-checked="true"],
    div[data-testid="stButtonGroup"] button[data-checked="true"] {
        background-color: #16a34a !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    div[data-testid="stButtonGroup"] button[aria-checked="false"],
    div[data-testid="stButtonGroup"] button[data-checked="false"] {
        background-color: #f1f5f9 !important;
        color: #334155 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stButtonGroup"] button:hover {
        background-color: #e2e8f0 !important;
    }
    div[data-testid="stButtonGroup"] button[aria-checked="true"]:hover {
        background-color: #15803d !important;
    }

    /* Clean styling for table Detail buttons */
    button[kind="tertiary"] {
        color: #16a34a !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
    }
    button[kind="tertiary"]:hover {
        color: #15803d !important;
        text-decoration: underline !important;
    }
    </style>
    """)
    
    # Dua tab: Log Aktivitas Sistem (sesuai screenshot) & Database Telemetri
    tab_activity, tab_telemetry = st.tabs([
        "📋 Riwayat Aktivitas & Prediksi AI",
        "📊 Database Telemetri Sensor (40 Hari)"
    ])
    
    # =========================================================================
    # TAB 1: RIWAYAT AKTIVITAS & PREDIKSI MODEL AI (PERSIS SCREENSHOT USER)
    # =========================================================================
    with tab_activity:
        st.html("""
        <div style="margin-bottom: 18px;">
            <h1 style="color: #0f172a; font-size: 2rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -0.5px;">
                History
            </h1>
            <p style="color: #64748b; font-size: 0.95rem; margin: 0; line-height: 1.5;">
                Riwayat aktivitas dan hasil prediksi model AI.
            </p>
        </div>
        """)
        
        # Container Filter Aktivitas
        st.html("""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px 20px 10px 20px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
            <div style="display: flex; align-items: center; gap: 8px; font-size: 0.88rem; font-weight: 600; color: #475569; margin-bottom: 8px;">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon>
                </svg>
                <span>Filter Aktivitas</span>
            </div>
        </div>
        """)
        
        # Streamlit Pills Filter
        filter_options = ["Semua", "LSTM", "CNN", "Pengaturan", "Sistem"]
        sel_filter = st.pills(
            "Filter Aktivitas:",
            filter_options,
            default="Semua",
            selection_mode="single",
            label_visibility="collapsed",
            key="activity_filter_pills"
        )
        if not sel_filter:
            sel_filter = "Semua"
            
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        # Load activity logs
        logs_df = get_activity_logs(sel_filter)
        
        # Container Tabel Aktivitas
        if logs_df.empty:
            st.html(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 36px 20px; text-align: center; color: #64748b;">
                <div style="font-size: 2rem; margin-bottom: 8px;">📭</div>
                <div style="font-size: 1rem; font-weight: 700; color: #1e293b;">Tidak Ada Riwayat untuk Filter "{sel_filter}"</div>
                <div style="font-size: 0.85rem; margin-top: 4px;">Belum ada aktivitas tercatat pada kategori ini.</div>
            </div>
            """)
        else:
            # Table Header
            st.html("""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-bottom: none; border-radius: 12px 12px 0 0; padding: 14px 20px; margin-top: 10px;">
                <div style="display: grid; grid-template-columns: 1.6fr 1.3fr 3.2fr 1.6fr 1.0fr; gap: 12px; align-items: center;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">WAKTU</div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">JENIS</div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">PROSES</div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">STATUS</div>
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">AKSI</div>
                </div>
            </div>
            """)
            
            # Rows rendering
            for idx, row in logs_df.iterrows():
                jenis_val = row['jenis']
                
                # Badge JENIS
                if jenis_val == "LSTM":
                    badge_jenis = """<span style="background: #eff6ff; color: #2563eb; font-weight: 700; font-size: 0.76rem; padding: 4px 12px; border-radius: 9999px; border: 1px solid #dbeafe; display: inline-flex; align-items: center; gap: 5px;"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.5"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"></polyline><polyline points="16 7 22 7 22 13"></polyline></svg> LSTM</span>"""
                elif jenis_val == "CNN":
                    badge_jenis = """<span style="background: #faf5ff; color: #9333ea; font-weight: 700; font-size: 0.76rem; padding: 4px 12px; border-radius: 9999px; border: 1px solid #f3e8ff; display: inline-flex; align-items: center; gap: 5px;"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#9333ea" stroke-width="2"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"></rect><circle cx="9" cy="9" r="2"></circle><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"></path></svg> CNN</span>"""
                elif jenis_val == "Pengaturan":
                    badge_jenis = """<span style="background: #fff7ed; color: #ea580c; font-weight: 700; font-size: 0.76rem; padding: 4px 12px; border-radius: 9999px; border: 1px solid #ffedd5; display: inline-flex; align-items: center; gap: 5px;"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#ea580c" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg> Pengaturan</span>"""
                else: # Sistem
                    badge_jenis = """<span style="background: #f1f5f9; color: #475569; font-weight: 700; font-size: 0.76rem; padding: 4px 12px; border-radius: 9999px; border: 1px solid #e2e8f0; display: inline-flex; align-items: center; gap: 5px;"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2"><rect width="20" height="14" x="2" y="3" rx="2"></rect><line x1="8" x2="16" y1="21" y2="21"></line><line x1="12" x2="12" y1="17" y2="21"></line></svg> Sistem</span>"""
                    
                # Badge STATUS
                badge_status = """<span style="background: #dcfce7; color: #16a34a; font-weight: 700; font-size: 0.76rem; padding: 4px 12px; border-radius: 9999px; border: 1px solid #bbf7d0; display: inline-flex; align-items: center; gap: 4px;">✔ Berhasil</span>"""
                
                col_w, col_j, col_p, col_s, col_a = st.columns([1.6, 1.3, 3.2, 1.6, 1.0], gap="small")
                with col_w:
                    st.markdown(f"<div style='font-size: 0.88rem; color: #475569; padding: 8px 0;'>{row['waktu']}</div>", unsafe_allow_html=True)
                with col_j:
                    st.html(f"<div style='padding: 6px 0;'>{badge_jenis}</div>")
                with col_p:
                    st.markdown(f"<div style='font-size: 0.88rem; font-weight: 500; color: #1e293b; padding: 8px 0;'>{row['proses']}</div>", unsafe_allow_html=True)
                with col_s:
                    st.html(f"<div style='padding: 6px 0;'>{badge_status}</div>")
                with col_a:
                    if st.button("Detail", key=f"btn_det_{row['id']}", type="tertiary"):
                        show_activity_detail(row)
                        
                st.markdown("<hr style='margin: 0; border: none; border-top: 1px solid #f1f5f9;'/>", unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: DATABASE TELEMETRI SENSOR (40 HARI LENGKAP)
    # =========================================================================
    with tab_telemetry:
        st.html("""
        <div class="hero-container" style="background: linear-gradient(135deg, #064e3b 0%, #0369a1 50%, #059669 100%); margin-bottom: 20px;">
            <div class="hero-badge">📜 Database Telemetri & Log Observasi</div>
            <h2 class="hero-title" style="font-size: 1.6rem;">Histori Observasi Pertumbuhan Pakcoy</h2>
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

        # 1. RINGKASAN STATISTIK DATASET 1.600+ DATA
        st.markdown("<h4 style='color: #065f46; margin-bottom: 12px;'>📊 Ringkasan Dataset Telemetri 40 Hari</h4>", unsafe_allow_html=True)
        
        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        
        with col_s1:
            st.html(f"""
            <div style="background: #ffffff; border: 1.5px solid #a7f3d0; border-radius: 12px; padding: 12px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
                <div style="font-size: 0.72rem; font-weight: 800; color: #047857; text-transform: uppercase;">TOTAL REKAMAN DATA</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #065f46; margin-top: 4px;">{total_records:,}</div>
                <div style="font-size: 0.72rem; color: #059669; font-weight: 600; margin-top: 2px;">≥ 1.600 Baris Data</div>
            </div>
            """)
            
        with col_s2:
            max_day = int(obs_df['day'].max())
            st.html(f"""
            <div style="background: #ffffff; border: 1.5px solid #bfdbfe; border-radius: 12px; padding: 12px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
                <div style="font-size: 0.72rem; font-weight: 800; color: #0284c7; text-transform: uppercase;">RENTANG BUDIDAYA</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #0369a1; margin-top: 4px;">{max_day} Hari</div>
                <div style="font-size: 0.72rem; color: #0284c7; font-weight: 600; margin-top: 2px;">Siklus Penuh 40 HST</div>
            </div>
            """)
            
        with col_s3:
            avg_moist = float(obs_df['soil_moisture'].mean())
            st.html(f"""
            <div style="background: #ffffff; border: 1.5px solid #bbf7d0; border-radius: 12px; padding: 12px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
                <div style="font-size: 0.72rem; font-weight: 800; color: #059669; text-transform: uppercase;">RATA-RATA KELEMBABAN</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #047857; margin-top: 4px;">{avg_moist:.1f}%</div>
                <div style="font-size: 0.72rem; color: #166534; font-weight: 600; margin-top: 2px;">Target: 50% - 65%</div>
            </div>
            """)
            
        with col_s4:
            avg_leaf = float(obs_df['leaf_width'].mean()) if 'leaf_width' in obs_df.columns else 6.5
            st.html(f"""
            <div style="background: #ffffff; border: 1.5px solid #ddd6fe; border-radius: 12px; padding: 12px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
                <div style="font-size: 0.72rem; font-weight: 800; color: #7c3aed; text-transform: uppercase;">RATA-RATA LEBAR DAUN</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #6d28d9; margin-top: 4px;">{avg_leaf:.1f} cm</div>
                <div style="font-size: 0.72rem; color: #7c3aed; font-weight: 600; margin-top: 2px;">Panen: ≥ 10.0 cm</div>
            </div>
            """)
            
        with col_s5:
            pump_on_count = int((obs_df['pump_status'] == 'ON').sum()) if 'pump_status' in obs_df.columns else 0
            st.html(f"""
            <div style="background: #ffffff; border: 1.5px solid #fed7aa; border-radius: 12px; padding: 12px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
                <div style="font-size: 0.72rem; font-weight: 800; color: #c2410c; text-transform: uppercase;">AKTIVASI POMPA AIR</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #ea580c; margin-top: 4px;">{pump_on_count:,}x</div>
                <div style="font-size: 0.72rem; color: #c2410c; font-weight: 600; margin-top: 2px;">Irigasi Otomatis IoT</div>
            </div>
            """)

        st.markdown("<br/>", unsafe_allow_html=True)
        
        # 2. FILTER DATA INTERAKTIF
        st.markdown("<h4 style='color: #065f46;'>🔍 Filter Log Pengamatan Telemetri</h4>", unsafe_allow_html=True)
        
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

        # 3. DOWNLOAD CSV & TABLE DISPLAY
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

        # Format table for presentation
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
            height=480
        )
