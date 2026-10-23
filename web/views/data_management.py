import streamlit as st
import pandas as pd
from services.data_service import get_all_plants, get_all_observations_df, add_observation_record

def render_data_management():
    st.html("""
    <div class="hero-container" style="background: linear-gradient(135deg, #064e3b 0%, #0369a1 50%, #059669 100%);">
        <div class="hero-badge">💾 Data Management</div>
        <h1 class="hero-title">Manajemen Dataset & Log Observasi</h1>
        <p class="hero-subtitle">
            Akses riwayat pengamatan deret waktu lengkap, lakukan entri observasi harian baru, 
            dan ekspor dataset terverifikasi dalam format CSV.
        </p>
    </div>
    """)
    
    tab_d1, tab_d2, tab_d3 = st.tabs(["📊 Tabel Log Observasi", "✍️ Entri Data Baru", "📥 Ekspor Dataset CSV"])
    
    with tab_d1:
        st.markdown("<h3 style='color: #065f46;'>Data Pengamatan Time-Series Terkini</h3>", unsafe_allow_html=True)
        obs_df = get_all_observations_df()
        
        f_c1, f_c2 = st.columns(2)
        with f_c1:
            plants_list = ["Semua"] + obs_df['plant_id'].unique().tolist()
            sel_plant = st.selectbox("Filter Berdasarkan Tanaman", plants_list)
        with f_c2:
            time_list = ["Semua", "08:00", "12:00", "16:00"]
            sel_time = st.selectbox("Filter Sesi Waktu", time_list)
            
        view_df = obs_df.copy()
        if sel_plant != "Semua":
            view_df = view_df[view_df['plant_id'] == sel_plant]
        if sel_time != "Semua":
            view_df = view_df[view_df['time'] == sel_time]
            
        st.write(f"Menampilkan **{len(view_df)}** baris data:")
        st.dataframe(view_df, use_container_width=True, hide_index=True)
        
    with tab_d2:
        st.markdown("<h3 style='color: #065f46;'>Formulir Entri Observasi Baru</h3>", unsafe_allow_html=True)
        plants_df = get_all_plants()
        
        with st.form("new_observation_form"):
            in_plant = st.selectbox("ID Tanaman", plants_df['plant_id'].tolist())
            in_day = st.number_input("Hari Pengamatan (Day)", min_value=1, max_value=500, value=36)
            in_dap = st.number_input("Umur Tanaman (HST / DAP)", min_value=1, max_value=550, value=45)
            in_time = st.selectbox("Waktu Pengamatan", ["08:00", "12:00", "16:00"])
            in_moist = st.number_input("Kelembaban Media Tanam (%)", min_value=10.0, max_value=100.0, value=52.0)
            in_temp = st.number_input("Suhu Lingkungan (°C)", min_value=15.0, max_value=45.0, value=26.5)
            
            in_condition = "Optimal"
            if in_moist < 50.0:
                in_condition = "Dry"
            elif in_moist > 75.0:
                in_condition = "Wet"
                
            in_maturity = st.number_input("Ground Truth Maturity Level (%)", min_value=1.0, max_value=100.0, value=38.0)
            in_criteria = st.selectbox("Kriteria Fase Pertumbuhan", [
                "Initial Stage",
                "Vegetative Stage",
                "Pre-Harvest Stage",
                "Harvest Ready"
            ])
            
            submit_btn = st.form_submit_button("💾 Simpan Data Observasi", type="primary")
            if submit_btn:
                success = add_observation_record(
                    in_plant, in_day, in_dap, in_time, in_moist, in_temp, in_condition, in_maturity, in_criteria
                )
                if success:
                    st.success(f"Data pengamatan untuk {in_plant} berhasil disimpan ke database!")
                    st.rerun()
                    
    with tab_d3:
        st.markdown("<h3 style='color: #065f46;'>Ekspor Data Sistem ke Format CSV</h3>", unsafe_allow_html=True)
        st.html("""
        <div class="plant-card" style="border-left: 5px solid #0284c7; padding: 20px;">
            <p style="color: #475569; margin: 0 0 14px 0;">
                Unduh seluruh dataset pengamatan sensorik terstruktur (120 baris observasi aktual, siklus budidaya 40 hari) untuk kebutuhan publikasi ilmiah, 
                pelaporan agribisnis, atau pengujian model lanjutan.
            </p>
        </div>
        """)
        
        full_csv = get_all_observations_df().to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Dataset Observasi Lengkap (CSV)",
            data=full_csv,
            file_name="dataset_observasi_pakcoy.csv",
            mime="text/csv",
            type="primary"
        )
