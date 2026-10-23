import streamlit as st
import pandas as pd
from services.data_service import get_all_plants, get_latest_observation, get_alerts
from services.anomaly_service import detect_anomalies_for_plant
from web.image_utils import get_image_base64
from web.views.plant_detail import render_plant_detail

def render_monitoring():
    # If drilldown into plant detail is active
    if st.session_state.get("view_drilldown", False):
        c_back, _ = st.columns([1, 4])
        with c_back:
            if st.button("← Kembali ke Daftar Tanaman", type="secondary"):
                st.session_state["view_drilldown"] = False
                st.rerun()
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        render_plant_detail(st.session_state["selected_plant_id"])
        return

    st.html("""
    <div class="hero-container" style="background: linear-gradient(135deg, #064e3b 0%, #047857 50%, #059669 100%);">
        <div class="hero-badge">📋 Direktori Pemantauan</div>
        <h1 class="hero-title">Katalog & Monitoring Tanaman Pakcoy</h1>
        <p class="hero-subtitle">Pantau seluruh batch budidaya secara simultan, amati status fisiologis daun, dinamika kelembaban media, dan peramalan panen.</p>
    </div>
    """)
    
    plants_df = get_all_plants()
    
    # Filter & Search Controls (Friendly & Intuitive)
    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        search_query = st.text_input("🔍 Cari ID Tanaman / Batch / Varietas", placeholder="Contoh: PCK-001 atau Nauli")
    with col_s2:
        status_filter = st.selectbox("Filter Status", ["Semua", "Optimal", "Perlu Perhatian"])
    with col_s3:
        sort_by = st.selectbox("Urutkan Berdasarkan", ["ID Tanaman", "HST Tertinggi", "Kematangan Tertinggi"])
        
    filtered_df = plants_df.copy()
    if search_query:
        filtered_df = filtered_df[
            filtered_df['plant_id'].str.contains(search_query, case=False) |
            filtered_df['name'].str.contains(search_query, case=False) |
            filtered_df['batch'].str.contains(search_query, case=False)
        ]
    if status_filter != "Semua":
        filtered_df = filtered_df[filtered_df['status'] == status_filter]
        
    st.markdown(f"<p style='color: #065f46; font-weight: 700; margin-bottom: 16px;'>Menampilkan {len(filtered_df)} unit budidaya pakcoy:</p>", unsafe_allow_html=True)
    
    grid_cols = st.columns(2)
    
    for idx, (_, plant) in enumerate(filtered_df.iterrows()):
        col = grid_cols[idx % 2]
        p_id = plant['plant_id']
        latest = get_latest_observation(p_id)
        anomalies = detect_anomalies_for_plant(p_id)
        img_b64 = get_image_base64(p_id)
        
        status_color = "status-optimal" if plant['status'] == "Optimal" else "status-warning"
        mat_pct = latest['maturity_level'] if latest else 0.0
        dap_val = latest['dap'] if latest else 0
        moist_val = latest['soil_moisture'] if latest else 0.0
        temp_val = latest['temperature'] if latest else 0.0
        criteria_label = latest['criteria'].split(' (')[0] if latest else '-'
        
        is_focused = (st.session_state.get("selected_plant_id") == p_id)
        card_border = "border: 2px solid #10b981; box-shadow: 0 8px 24px -4px rgba(16, 185, 129, 0.2);" if is_focused else "border: 1px solid #e2e8f0;"
        focus_badge = "<span style='background: #ecfdf5; color: #047857; font-size: 0.72rem; font-weight: 800; padding: 2px 8px; border-radius: 9999px; border: 1px solid #a7f3d0;'>● FOKUS AKTIF</span>" if is_focused else ""
        
        # Harvest indicator calculations
        if dap_val >= 40:
            harvest_badge_html = f"""
            <div style="background: #fee2e2; color: #991c1c; border: 1px solid #fca5a5; font-size: 0.78rem; font-weight: 700; padding: 6px 10px; border-radius: 8px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
                <span>🚨 Batas Akhir Panen Tercapai ({dap_val}/40 HST)</span>
                <strong style="background: #b91c1c; color: #ffffff; padding: 1px 6px; border-radius: 4px; font-size: 0.72rem;">Wajib Panen</strong>
            </div>
            """
        elif dap_val >= 35:
            harvest_badge_html = f"""
            <div style="background: #fef3c7; color: #92400e; border: 1px solid #fde68a; font-size: 0.78rem; font-weight: 700; padding: 6px 10px; border-radius: 8px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
                <span>🥬 Jendela Panen Prima ({dap_val}/40 HST)</span>
                <strong style="color: #b45309;">Sisa {40 - dap_val} Hari</strong>
            </div>
            """
        else:
            harvest_badge_html = f"""
            <div style="background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; font-size: 0.78rem; font-weight: 700; padding: 6px 10px; border-radius: 8px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
                <span>🌱 Siklus Pertumbuhan ({dap_val}/40 HST)</span>
                <strong style="color: #047857;">Sisa {40 - dap_val} Hari ke Panen</strong>
            </div>
            """
        
        with col:
            st.html(f"""
            <div class="plant-card" style="{card_border} padding: 18px; margin-bottom: 16px;">
                <div style="display: flex; gap: 14px; margin-bottom: 12px; align-items: center;">
                    <img src="{img_b64}" style="width: 85px; height: 85px; border-radius: 12px; object-fit: cover; border: 1.5px solid #d1fae5;" />
                    <div style="flex: 1;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 0.76rem; font-weight: 800; color: #047857; background: #ecfdf5; border: 1px solid #a7f3d0; padding: 2px 8px; border-radius: 6px;">
                                {p_id}
                            </span>
                            <span class="status-badge {status_color}">{plant['status']}</span>
                        </div>
                        <h3 style="margin: 4px 0 2px 0; color: #0f172a; font-size: 1.15rem; font-weight: 800;">{plant['name']}</h3>
                        <p style="margin: 0; color: #64748b; font-size: 0.8rem;">
                            {plant['variety']} • {plant['batch']}
                        </p>
                        <div style="margin-top: 4px;">{focus_badge}</div>
                    </div>
                </div>
                
                {harvest_badge_html}
                
                <div style="margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 700; margin-bottom: 4px;">
                        <span style="color: #475569;">Indeks Kematangan ({criteria_label})</span>
                        <span style="color: #047857; font-weight: 800;">{mat_pct:.1f}%</span>
                    </div>
                    <div style="background: #e2e8f0; border-radius: 9999px; height: 8px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #10b981 0%, #059669 100%); width: {min(mat_pct, 100)}%; height: 100%; border-radius: 9999px;"></div>
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; background: #f8fafc; border: 1px solid #e2e8f0; padding: 10px; border-radius: 12px; margin-bottom: 14px; text-align: center;">
                    <div>
                        <span style="font-size: 0.7rem; color: #64748b; font-weight: 600; display: block;">Umur / Panen</span>
                        <strong style="color: {'#dc2626' if dap_val >= 40 else '#047857'}; font-size: 0.88rem;">{dap_val}/40 HST</strong>
                    </div>
                    <div>
                        <span style="font-size: 0.7rem; color: #64748b; font-weight: 600; display: block;">Kelembaban</span>
                        <strong style="color: #0284c7; font-size: 0.88rem;">{moist_val:.1f}%</strong>
                    </div>
                    <div>
                        <span style="font-size: 0.7rem; color: #64748b; font-weight: 600; display: block;">Kondisi</span>
                        <strong style="color: #059669; font-size: 0.88rem;">{latest['soil_condition'] if latest else '-'}</strong>
                    </div>
                    <div>
                        <span style="font-size: 0.7rem; color: #64748b; font-weight: 600; display: block;">Suhu</span>
                        <strong style="color: #d97706; font-size: 0.88rem;">{temp_val:.1f}°C</strong>
                    </div>
                </div>
            </div>
            """)
            
            c_btn1, c_btn2 = st.columns([1, 1.2])
            with c_btn1:
                if st.button(f"🎯 Fokus Tanaman", key=f"sel_{p_id}", use_container_width=True):
                    st.session_state["selected_plant_id"] = p_id
                    st.rerun()
            with c_btn2:
                if st.button(f"🌿 Detail & Timeline →", key=f"drill_{p_id}", type="primary", use_container_width=True):
                    st.session_state["selected_plant_id"] = p_id
                    st.session_state["view_drilldown"] = True
                    st.rerun()
                    
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
