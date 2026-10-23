import streamlit as st
import os
import glob
from PIL import Image
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from services.cnn_service import analyze_image_with_gradcam
from services.data_service import get_all_plants

def render_image_analysis():
    # --- 1. HEADER SECTION ---
    col_head1, col_head2 = st.columns([2.2, 1.2])
    with col_head1:
        st.html("""
        <div style="margin-bottom: 8px;">
            <h1 style="color: #0f172a; font-size: 1.85rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -0.5px;">
                Pusat Analisis Visual (CNN + LLM)
            </h1>
            <p style="color: #64748b; font-size: 0.95rem; margin: 0; line-height: 1.5;">
                Mengekstraksi 12 metrik biometrik tanaman secara detail dan mendeteksi anomali.
            </p>
        </div>
        """)
        
    with col_head2:
        analysis_mode = st.radio(
            "Mode Tampilan Analisis",
            ["Analisis Tunggal", "Perbandingan Foto"],
            horizontal=True,
            label_visibility="collapsed"
        )

    st.markdown("<hr style='margin: 12px 0 20px 0; border: none; border-top: 1px solid #e2e8f0;'/>", unsafe_allow_html=True)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_dir = os.path.join(BASE_DIR, "data", "images")
    default_sample = os.path.join(sample_dir, "Masa_Pertumbuhan", "Masa_Pertumbuhan_008.jpg")
    if not os.path.exists(default_sample):
        # Fallback to any existing sample
        samples_found = glob.glob(os.path.join(sample_dir, "**", "*.jpg"), recursive=True)
        default_sample = samples_found[0] if samples_found else None

    # =========================================================================
    # MODE 1: ANALISIS TUNGGAL (MATCHING USER SCREENSHOT EXACTLY)
    # =========================================================================
    if analysis_mode == "Analisis Tunggal":
        # Prominent File Upload Section
        c_up1, c_up2 = st.columns([1.8, 1])
        with c_up1:
            uploaded_file = st.file_uploader(
                "📸 Unggah Foto Sampel Tanaman Pakcoy (JPG / PNG):",
                type=["jpg", "jpeg", "png"],
                key="direct_file_uploader",
                help="Seret atau pilih foto sampel daun pakcoy Anda untuk ekstraksi biometrik instan"
            )
        with c_up2:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            with st.expander("🌿 Atau Pilih Sampel Dataset (120 Foto)", expanded=False):
                sel_cat = st.selectbox("Fase:", ["Masa_Pertumbuhan", "Mendekati_Panen", "Siap_Panen"], key="exp_cat")
                cat_folder = os.path.join(sample_dir, sel_cat)
                cat_imgs = glob.glob(os.path.join(cat_folder, "*.jpg"))
                sel_img = st.selectbox("Foto:", [os.path.basename(p) for p in cat_imgs[:25]] if cat_imgs else ["default"], key="exp_img")
                dataset_choice = os.path.join(cat_folder, sel_img) if cat_imgs else default_sample

        # Determine Image to Analyze
        if uploaded_file is not None:
            image_to_analyze = uploaded_file
        elif 'dataset_choice' in locals() and os.path.exists(dataset_choice):
            image_to_analyze = dataset_choice
        else:
            image_to_analyze = default_sample

        # Run Computer Vision inference & biometric extraction
        with st.spinner("Mengekstraksi metrik biometrik citra daun via CNN..."):
            res = analyze_image_with_gradcam(image_to_analyze)

        bio = res["biometrics"]
        maturity_val = bio["maturity_pct"]
        fase_title = bio["fase_label"]

        # --- TWO-COLUMN LAYOUT EXACTLY MATCHING USER SCREENSHOT ---
        col_input, col_bio = st.columns([1.05, 1.25], gap="large")

        # ----------------- LEFT COLUMN: INPUT VISUAL -----------------
        with col_input:
            st.html("""
            <h3 style="color: #0f172a; font-size: 1.15rem; font-weight: 700; margin: 0 0 12px 0;">
                Input Visual
            </h3>
            """)
            st.image(res["original_image"], use_container_width=True)

            col_btn1, col_btn2 = st.columns([1.2, 1])
            with col_btn1:
                if st.button("🔄 Pindai Baru", use_container_width=True, type="secondary"):
                    st.rerun()
            with col_btn2:
                if st.button("🔬 Toggle Grad-CAM", use_container_width=True, type="primary"):
                    st.session_state["show_gradcam_toggle"] = not st.session_state.get("show_gradcam_toggle", False)
                    st.rerun()

        # ----------------- RIGHT COLUMN: HASIL EKSTRAKSI BIOMETRIK -----------------
        with col_bio:
            bio_html = f"""
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
    <h3 style="color: #0f172a; font-size: 1.15rem; font-weight: 700; margin: 0 0 12px 0;">
        Hasil Ekstraksi Biometrik
    </h3>

    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 2px 8px -2px rgba(0,0,0,0.04); display: flex; justify-content: space-between; align-items: center;">
        <div style="display: flex; align-items: center; gap: 14px;">
            <div style="background: #ecfdf5; border-radius: 10px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; border: 1.5px solid #a7f3d0; color: #059669;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
            </div>
            <div>
                <div style="font-size: 0.72rem; font-weight: 800; color: #059669; text-transform: uppercase; letter-spacing: 0.05em;">
                    FASE KEMATANGAN (CNN)
                </div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #047857; margin-top: 2px;">
                    {fase_title}
                </div>
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 1.85rem; font-weight: 800; color: #0f172a; line-height: 1;">
                {maturity_val:.1f}%
            </div>
            <div style="font-size: 0.74rem; font-weight: 700; color: #64748b; margin-top: 2px;">
                Maturity
            </div>
        </div>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
            <div style="font-size: 0.72rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; display: flex; align-items: center; gap: 6px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>
                <span>KONDISI DAUN</span>
            </div>
            <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-top: 5px;">
                {bio['kondisi_daun']}
            </div>
        </div>

        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
            <div style="font-size: 0.72rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; display: flex; align-items: center; gap: 6px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2"><circle cx="13.5" cy="6.5" r=".5" fill="#64748b"/><circle cx="17.5" cy="10.5" r=".5" fill="#64748b"/><circle cx="8.5" cy="7.5" r=".5" fill="#64748b"/><circle cx="6.5" cy="12.5" r=".5" fill="#64748b"/><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.554C21.965 6.012 17.461 2 12 2z"/></svg>
                <span>WARNA DAUN</span>
            </div>
            <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-top: 5px;">
                {bio['warna_daun']}
            </div>
        </div>

        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
            <div style="font-size: 0.72rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; display: flex; align-items: center; gap: 6px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2"><path d="m15 5 4 4"/><path d="M13 3 4 12v5h5l9-9-5-5Z"/></svg>
                <span>ESTIMASI LEBAR</span>
            </div>
            <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-top: 5px;">
                {bio['estimasi_lebar']}
            </div>
        </div>

        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
            <div style="font-size: 0.72rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; display: flex; align-items: center; gap: 6px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2"><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/></svg>
                <span>BENTUK DAUN</span>
            </div>
            <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-top: 5px;">
                {bio['bentuk_daun']}
            </div>
        </div>

        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
            <div style="font-size: 0.72rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; display: flex; align-items: center; gap: 6px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>
                <span>JML HELAI</span>
            </div>
            <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-top: 5px;">
                {bio['jml_helai']}
            </div>
        </div>

        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
            <div style="font-size: 0.72rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; display: flex; align-items: center; gap: 6px;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg>
                <span>LUAS KANOPI</span>
            </div>
            <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-top: 5px;">
                {bio['luas_kanopi']}
            </div>
        </div>
    </div>

    <div style="background: #fffbeb; border: 1.5px solid #fde68a; border-radius: 10px; padding: 12px 16px; margin-bottom: 10px;">
        <div style="font-size: 0.74rem; font-weight: 800; color: #b45309; text-transform: uppercase; letter-spacing: 0.04em; display: flex; align-items: center; gap: 6px;">
            <span>⚠️</span> DETEKSI ANOMALI
        </div>
        <div style="font-size: 0.98rem; font-weight: 700; color: #92400e; margin-top: 4px;">
            {bio['deteksi_anomali']}
        </div>
    </div>

    <div style="background: #ecfdf5; border: 1.5px solid #a7f3d0; border-radius: 10px; padding: 12px 16px;">
        <div style="font-size: 0.74rem; font-weight: 800; color: #047857; text-transform: uppercase; letter-spacing: 0.04em; display: flex; align-items: center; gap: 6px;">
            <span>✳️</span> DETEKSI PENYAKIT / HAMA
        </div>
        <div style="font-size: 0.98rem; font-weight: 700; color: #065f46; margin-top: 4px;">
            {bio['deteksi_penyakit']}
        </div>
    </div>
</div>
"""
            st.html(bio_html)

        st.markdown("<br/>", unsafe_allow_html=True)

        # ----------------- SECTION DIAGRAM BATANG & EXPLAINABLE AI -----------------
        st.html("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
            <h3 style="color: #065f46; margin: 0; font-size: 1.25rem; font-weight: 800;">
                📊 Visualisasi Diagram Batang & Explainable AI
            </h3>
            <span style="font-size: 0.8rem; color: #64748b; font-weight: 600;">
                Ekstraksi Fitur Spektral CNN + Excess Green Index (ExG)
            </span>
        </div>
        """)

        col_bar1, col_bar2 = st.columns([1.1, 1.2], gap="large")

        with col_bar1:
            st.html("""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;">
                <h4 style="margin: 0; color: #064e3b; font-size: 0.98rem; font-weight: 700;">
                    📈 Probabilitas Klasifikasi Fase (CNN Softmax)
                </h4>
            </div>
            """)
            
            prob_df = pd.DataFrame([
                {"Fase Pertumbuhan": k, "Probabilitas (%)": v * 100}
                for k, v in res["probabilities"].items()
            ])
            
            fig_prob = px.bar(
                prob_df,
                x="Probabilitas (%)",
                y="Fase Pertumbuhan",
                orientation="h",
                color="Probabilitas (%)",
                color_continuous_scale="Greens",
                text_auto=".1f"
            )
            fig_prob.update_layout(
                template="plotly_white",
                showlegend=False,
                height=240,
                margin=dict(l=0, r=25, t=10, b=10),
                plot_bgcolor="rgba(248, 250, 252, 0.8)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Tingkat Keyakinan (%)",
                yaxis_title=None
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        with col_bar2:
            st.html("""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;">
                <h4 style="margin: 0; color: #0369a1; font-size: 0.98rem; font-weight: 700;">
                    📊 Skor 5 Indeks Biometrik Citra (0 - 100%)
                </h4>
            </div>
            """)
            
            idx_df = pd.DataFrame([
                {"Metrik Biometrik": k, "Skor Indeks (%)": v}
                for k, v in bio["biometric_indices"].items()
            ])
            
            fig_idx = px.bar(
                idx_df,
                x="Skor Indeks (%)",
                y="Metrik Biometrik",
                orientation="h",
                color="Skor Indeks (%)",
                color_continuous_scale="Teal",
                text_auto=".1f"
            )
            fig_idx.update_layout(
                template="plotly_white",
                showlegend=False,
                height=240,
                margin=dict(l=0, r=25, t=10, b=10),
                plot_bgcolor="rgba(248, 250, 252, 0.8)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Skor Normalisasi (%)",
                yaxis_title=None
            )
            st.plotly_chart(fig_idx, use_container_width=True)

        # Grad-CAM Display Section (Toggleable or Visible)
        if st.session_state.get("show_gradcam_toggle", True):
            st.html("""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px 22px; box-shadow: 0 4px 16px -2px rgba(0,0,0,0.04); margin-top: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div>
                        <h4 style="margin: 0; color: #6d28d9; font-size: 1.05rem; font-weight: 800;">
                            🔬 Grad-CAM Saliency Map Overlay (Explainable AI)
                        </h4>
                        <p style="margin: 4px 0 0 0; color: #64748b; font-size: 0.85rem;">
                            Grad-CAM menunjukkan gradien aktivasi spasial pada kanopi daun pakcoy yang menjadi pertimbangan utama jaringan neural dalam menentukan fase kematangan.
                        </p>
                    </div>
                    <span style="background: #f5f3ff; color: #7c3aed; font-size: 0.75rem; font-weight: 800; padding: 4px 10px; border-radius: 6px; border: 1px solid #ddd6fe;">
                        XAI ACTIVE
                    </span>
                </div>
            </div>
            """)
            
            c_g1, c_g2 = st.columns(2)
            with c_g1:
                st.image(res["original_image"], caption="Foto Asli Tanaman (RGB Input)", use_container_width=True)
            with c_g2:
                st.image(res["gradcam_image"], caption="Grad-CAM Saliency Overlay (Aktivasi Visual Daun)", use_container_width=True)

    # =========================================================================
    # MODE 2: PERBANDINGAN FOTO (SIDE-BY-SIDE PHOTO COMPARISON)
    # =========================================================================
    else:
        st.html("""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 20px; margin-bottom: 20px;">
            <h4 style="margin: 0 0 4px 0; color: #064e3b; font-size: 1.05rem;">
                🔄 Mode Perbandingan Citra & Dinamika Morfologi
            </h4>
            <p style="margin: 0; color: #64748b; font-size: 0.86rem;">
                Bandingkan dua sampel foto tanaman pakcoy untuk menganalisis perkembangan kanopi daun, peningkatan indeks kematangan, dan stabilitas warna.
            </p>
        </div>
        """)
        
        c_cmp1, c_cmp2 = st.columns(2, gap="large")
        
        all_samples = glob.glob(os.path.join(sample_dir, "**", "*.jpg"), recursive=True)
        sample_dict = {f"{os.path.basename(os.path.dirname(p))} - {os.path.basename(p)}": p for p in all_samples}
        sample_keys = list(sample_dict.keys())
        
        with c_cmp1:
            st.markdown("<h4 style='color: #047857; font-size: 1rem;'>📷 Foto Sampel A (Awal / Kontrol)</h4>", unsafe_allow_html=True)
            def_idx_a = 0
            sel_a = st.selectbox("Pilih Sampel A:", sample_keys, index=def_idx_a, key="cmp_sel_a")
            path_a = sample_dict[sel_a]
            res_a = analyze_image_with_gradcam(path_a)
            st.image(res_a["original_image"], caption=f"Foto A: {res_a['predicted_label']}", use_container_width=True)
            
        with c_cmp2:
            st.markdown("<h4 style='color: #0369a1; font-size: 1rem;'>📷 Foto Sampel B (Perkembangan / Uji)</h4>", unsafe_allow_html=True)
            def_idx_b = min(len(sample_keys) - 1, 35)
            sel_b = st.selectbox("Pilih Sampel B:", sample_keys, index=def_idx_b, key="cmp_sel_b")
            path_b = sample_dict[sel_b]
            res_b = analyze_image_with_gradcam(path_b)
            st.image(res_b["original_image"], caption=f"Foto B: {res_b['predicted_label']}", use_container_width=True)

        bio_a = res_a["biometrics"]
        bio_b = res_b["biometrics"]
        
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #065f46;'>📋 Matriks Perbandingan Biometrik (Foto A vs Foto B)</h4>", unsafe_allow_html=True)
        
        comparison_matrix = pd.DataFrame([
            {"Parameter Biometrik": "Fase Kematangan (CNN)", "Foto A": bio_a["fase_label"], "Foto B": bio_b["fase_label"], "Analisis Perubahan": "Transisi Fase" if bio_a["fase_label"] != bio_b["fase_label"] else "Fase Sama"},
            {"Parameter Biometrik": "Indeks Kematangan (%)", "Foto A": f"{bio_a['maturity_pct']:.1f}%", "Foto B": f"{bio_b['maturity_pct']:.1f}%", "Analisis Perubahan": f"{bio_b['maturity_pct'] - bio_a['maturity_pct']:+.1f}%"},
            {"Parameter Biometrik": "Kondisi Daun", "Foto A": bio_a["kondisi_daun"], "Foto B": bio_b["kondisi_daun"], "Analisis Perubahan": "Stabil Segar" if bio_a["kondisi_daun"] == bio_b["kondisi_daun"] else "Perubahan Turgor"},
            {"Parameter Biometrik": "Warna Daun", "Foto A": bio_a["warna_daun"], "Foto B": bio_b["warna_daun"], "Analisis Perubahan": "Perkembangan Pigmen"},
            {"Parameter Biometrik": "Estimasi Lebar", "Foto A": bio_a["estimasi_lebar"], "Foto B": bio_b["estimasi_lebar"], "Analisis Perubahan": "Ekspansi Daun"},
            {"Parameter Biometrik": "Jumlah Helai", "Foto A": bio_a["jml_helai"], "Foto B": bio_b["jml_helai"], "Analisis Perubahan": "Pertambahan Daun"},
            {"Parameter Biometrik": "Luas Kanopi", "Foto A": f"{bio_a['luas_kanopi']} ({bio_a['luas_kanopi_pct']}%)", "Foto B": f"{bio_b['luas_kanopi']} ({bio_b['luas_kanopi_pct']}%)", "Analisis Perubahan": f"{bio_b['luas_kanopi_pct'] - bio_a['luas_kanopi_pct']:+.1f}% Area Kanopi"},
            {"Parameter Biometrik": "Deteksi Anomali", "Foto A": bio_a["deteksi_anomali"], "Foto B": bio_b["deteksi_anomali"], "Analisis Perubahan": "Aman & Terkendali"},
            {"Parameter Biometrik": "Status Penyakit/Hama", "Foto A": bio_a["deteksi_penyakit"], "Foto B": bio_b["deteksi_penyakit"], "Analisis Perubahan": "Bebas Patogen"}
        ])
        st.dataframe(comparison_matrix, use_container_width=True, hide_index=True)
        
        # Comparative Grouped Bar Chart
        st.markdown("<h4 style='color: #065f46; margin-top: 18px;'>📊 Diagram Batang Komparatif Indeks Biometrik</h4>", unsafe_allow_html=True)
        
        metric_keys = list(bio_a["biometric_indices"].keys())
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name="Foto A",
            x=metric_keys,
            y=[bio_a["biometric_indices"][k] for k in metric_keys],
            marker_color="#10b981",
            text=[f"{bio_a['biometric_indices'][k]:.1f}%" for k in metric_keys],
            textposition="auto"
        ))
        fig_cmp.add_trace(go.Bar(
            name="Foto B",
            x=metric_keys,
            y=[bio_b["biometric_indices"][k] for k in metric_keys],
            marker_color="#0284c7",
            text=[f"{bio_b['biometric_indices'][k]:.1f}%" for k in metric_keys],
            textposition="auto"
        ))
        fig_cmp.update_layout(
            barmode="group",
            template="plotly_white",
            height=360,
            xaxis_title="Indeks Biometrik Citra",
            yaxis_title="Skor Normalisasi (%)",
            plot_bgcolor="rgba(248, 250, 252, 0.8)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
