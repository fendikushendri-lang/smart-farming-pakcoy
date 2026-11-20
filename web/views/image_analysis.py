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

def render_image_analysis(selected_plant_id: str = "PCK-001"):
    # --- 1. HEADER SECTION ---
    col_head1, col_head2 = st.columns([2.2, 1.2])
    with col_head1:
        st.html("""
        <div style="margin-bottom: 8px;">
            <h1 style="color: #0f172a; font-size: 1.85rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -0.5px;">
                Pusat Analisis Visual Pakcoy (CNN + Vision AI)
            </h1>
            <p style="color: #64748b; font-size: 0.95rem; margin: 0; line-height: 1.5;">
                Verifikasi spesies tanaman otomatis (Pakcoy vs Selada vs Bayam vs Seledri), 
                penentuan kesiapan panen berdasarkan lebar daun (standar ≥ 10 cm), dan eksplanasi visual <em>Grad-CAM</em>.
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
    light_sample_dir = os.path.join(BASE_DIR, "data", "sample_images")

    # Pemetaan foto sampel pakcoy autentik sesuai tanaman/batch yang dipilih pengguna
    plant_authentic_samples = {
        "PCK-001": os.path.join(sample_dir, "Masa_Pertumbuhan", "Masa_Pertumbuhan_040_cotyledon.jpg"),
        "PCK-002": os.path.join(sample_dir, "Masa_Pertumbuhan", "Masa_Pertumbuhan_006.jpg"),
        "PCK-003": os.path.join(sample_dir, "Mendekati_Panen", "Mendekati_Panen_039.jpg"),
        "PCK-004": os.path.join(sample_dir, "Siap_Panen", "Siap_Panen_080.jpg"),
        "PCK-005": os.path.join(sample_dir, "Siap_Panen", "Siap_Panen_073.jpg"),
        "PCK-FULL40": os.path.join(sample_dir, "Siap_Panen", "Siap_Panen_080.jpg"),
    }
    
    # Default sample image dinamis sesuai umur HST batch yang dipilih (PCK-001 s.d. PCK-040)
    default_sample = plant_authentic_samples.get(selected_plant_id)
    if not default_sample or not os.path.isfile(default_sample):
        from services.data_service import get_latest_observation
        obs = get_latest_observation(selected_plant_id)
        dap_num = obs.get("dap", 20) if obs else 20
        if dap_num <= 4:
            cand = os.path.join(sample_dir, "Masa_Pertumbuhan", "Masa_Pertumbuhan_040_cotyledon.jpg")
        elif dap_num <= 25:
            cand = os.path.join(sample_dir, "Masa_Pertumbuhan", "Masa_Pertumbuhan_006.jpg")
        elif dap_num <= 34:
            cand = os.path.join(sample_dir, "Mendekati_Panen", "Mendekati_Panen_039.jpg")
        else:
            cand = os.path.join(sample_dir, "Siap_Panen", "Siap_Panen_080.jpg")
            
        if os.path.isfile(cand):
            default_sample = cand
        else:
            primary_sample = os.path.join(sample_dir, "Siap_Panen", "Siap_Panen_080.jpg")
            default_sample = primary_sample if os.path.isfile(primary_sample) else None

    # Kelompok folder sampel untuk memudahkan pengujian multi-spesies & fase pakcoy
    sample_groups = {
        "🌱 Pakcoy: Fase Pertumbuhan": os.path.join(sample_dir, "Masa_Pertumbuhan"),
        "🍃 Pakcoy: Mendekati Panen": os.path.join(sample_dir, "Mendekati_Panen"),
        "🥗 Pakcoy: Siap Panen": os.path.join(sample_dir, "Siap_Panen"),
        "🥬 Bibit Selada (Bukan Pakcoy)": os.path.join(sample_dir, "non_pakcoy", "Bibit_Selada"),
        "🥬 Daun Selada Dewasa (Bukan Pakcoy)": os.path.join(sample_dir, "non_pakcoy", "Daun_Selada"),
        "🌿 Bibit Bayam (Bukan Pakcoy)": os.path.join(sample_dir, "non_pakcoy", "Bibit_Bayam"),
        "🌿 Daun Bayam (Bukan Pakcoy)": os.path.join(sample_dir, "non_pakcoy", "Daun_Bayam"),
        "🌱 Seledri (Bukan Pakcoy)": os.path.join(sample_dir, "non_pakcoy", "Seledri"),
        "☘️ Daun Lainnya (Bukan Pakcoy)": os.path.join(sample_dir, "non_pakcoy", "Daun_Lainnya")
    }

    # =========================================================================
    # MODE 1: ANALISIS TUNGGAL
    # =========================================================================
    if analysis_mode == "Analisis Tunggal":
        c_up1, c_up2 = st.columns([1.8, 1])
        with c_up1:
            uploaded_file = st.file_uploader(
                "📸 Unggah Foto Sampel Tanaman (JPG / PNG):",
                type=["jpg", "jpeg", "png", "webp"],
                key="direct_file_uploader",
                help="Seret atau pilih foto daun pakcoy atau tanaman lain untuk dianalisis oleh CNN"
            )
        with c_up2:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            with st.expander("🌿 Atau Pilih Sampel Tanaman", expanded=False):
                sel_group_label = st.selectbox("Pilih Kategori Sampel:", list(sample_groups.keys()), key="exp_grp")
                cat_folder = sample_groups[sel_group_label]
                
                avail_files = [os.path.basename(p) for p in glob.glob(os.path.join(cat_folder, "*")) if p.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                dataset_choice = None
                if avail_files:
                    sel_img = st.selectbox("Pilih File Foto:", avail_files[:20], key="exp_img")
                    dataset_choice = os.path.join(cat_folder, sel_img)
                else:
                    st.caption("ℹ️ Sampel folder tidak ditemukan. Silakan unggah foto di samping.")

        # Determine Image to Analyze
        image_to_analyze = None
        if uploaded_file is not None:
            image_to_analyze = uploaded_file
        elif dataset_choice and isinstance(dataset_choice, str) and os.path.isfile(dataset_choice):
            image_to_analyze = dataset_choice
        elif default_sample and isinstance(default_sample, str) and os.path.isfile(default_sample):
            image_to_analyze = default_sample

        if image_to_analyze is None:
            st.html("""
            <div style="background: #f8fafc; border: 2px dashed #94a3b8; border-radius: 14px; padding: 36px 20px; text-align: center; margin-top: 18px;">
                <div style="font-size: 2.5rem; margin-bottom: 8px;">📷</div>
                <h3 style="color: #0f172a; font-size: 1.15rem; font-weight: 700; margin: 0 0 6px 0;">
                    Silakan Unggah Foto Sampel Tanaman
                </h3>
                <p style="color: #64748b; font-size: 0.92rem; max-width: 500px; margin: 0 auto; line-height: 1.5;">
                    Unggah foto tanaman Anda untuk identifikasi spesies otomatis (Pakcoy, Selada, Bayam, Seledri), penentuan lebar daun panen pakcoy, dan visualisasi aktivasi Grad-CAM.
                </p>
            </div>
            """)
            return

        # Run Hierarchical Computer Vision Inference
        with st.spinner("Memindai citra daun via CNN Hierarkis (Spesies + Morfologi)..."):
            res = analyze_image_with_gradcam(image_to_analyze)

        is_pakcoy = res.get("is_pakcoy", True)
        spec_info = res.get("species_info") or {}
        spec_name = res.get("species_name", "Bukan Tanaman Pakcoy")
        spec_conf = res.get("species_confidence", res.get("confidence", 90.0))
        bio = res["biometrics"]
        maturity_val = bio["maturity_pct"]
        fase_title = bio["fase_label"]

        # Log activity CNN
        img_key = str(image_to_analyze) if isinstance(image_to_analyze, str) else getattr(image_to_analyze, 'name', 'upload')
        if f"logged_cnn_{img_key}" not in st.session_state:
            try:
                from services.data_service import add_activity_log
                add_activity_log(
                    "CNN",
                    "CNN Image Analysis",
                    "Berhasil" if is_pakcoy else "Non-Pakcoy",
                    {
                        "spesies": spec_name,
                        "fase": res["predicted_label"],
                        "keyakinan": f"{res['confidence']:.1f}%",
                        "lebar_daun": f"{bio.get('leaf_width_cm', 0.0)} cm",
                        "sampel": os.path.basename(img_key)
                    }
                )
                st.session_state[f"logged_cnn_{img_key}"] = True
            except Exception:
                pass

        # Simpan hasil analisis CNN ke session_state untuk grounding Asisten AI
        st.session_state["latest_cnn_result"] = {
            "species": spec_name,
            "stage": res.get("predicted_label", "Masa Pertumbuhan"),
            "leaf_width": f"{bio.get('leaf_width_cm', 0.0):.1f}",
            "confidence": f"{res.get('confidence', 90.0):.1f}%",
            "status": "Siap Panen" if bio.get("harvest_ready") else "Masa Pertumbuhan"
        }

        # --- TWO-COLUMN LAYOUT ---
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
            if is_pakcoy:
                leaf_w_val = bio.get('leaf_width_cm', 10.5)
                is_ready = leaf_w_val >= 10.0
                badge_color = "#059669" if is_ready else ("#d97706" if leaf_w_val >= 6.0 else "#0284c7")
                badge_bg = "#ecfdf5" if is_ready else ("#fffbeb" if leaf_w_val >= 6.0 else "#eff6ff")
                badge_border = "#a7f3d0" if is_ready else ("#fde68a" if leaf_w_val >= 6.0 else "#bfdbfe")
                status_text = "SIAP PANEN (MATANG OPTIMAL)" if is_ready else ("MENDEKATI PANEN (PRA-PANEN)" if leaf_w_val >= 6.0 else "MASA PERTUMBUHAN (BELUM SIAP PANEN)")
                progress_pct = min(round((leaf_w_val / 10.0) * 100.0, 1), 100.0)

                # Dedicated Leaf-Width Harvest Determination Card
                st.html(f"""
                <div style="background: {badge_bg}; border: 1.5px solid {badge_border}; border-radius: 12px; padding: 14px 18px; margin-bottom: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 0.76rem; font-weight: 800; color: {badge_color}; text-transform: uppercase; letter-spacing: 0.05em;">
                            🍃 PENENTUAN MASA PANEN BERDASARKAN LEBAR DAUN (CNN)
                        </span>
                        <span style="background: {badge_color}; color: #ffffff; font-size: 0.72rem; font-weight: 800; padding: 2px 8px; border-radius: 6px;">
                            {status_text}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                        <div>
                            <span style="font-size: 1.9rem; font-weight: 800; color: #0f172a;">{leaf_w_val} cm</span>
                            <span style="font-size: 0.82rem; color: #64748b; margin-left: 6px;">(Standar Panen: ≥ 10.0 cm)</span>
                        </div>
                        <div style="font-size: 0.85rem; font-weight: 700; color: {badge_color};">
                            {bio.get('estimasi_sisa_hari', 'Siap Panen')}
                        </div>
                    </div>
                    <div style="background: #e2e8f0; border-radius: 9999px; height: 8px; overflow: hidden; margin: 8px 0 6px 0;">
                        <div style="background: {badge_color}; width: {progress_pct}%; height: 100%; border-radius: 9999px;"></div>
                    </div>
                    <p style="font-size: 0.8rem; color: #475569; margin: 4px 0 0 0; line-height: 1.4;">
                        💡 <strong>Kaidah Fisiologis:</strong> {bio.get('rekomendasi_panen', 'Pakcoy dipanen saat daun mangkok mencapai lebar standar 10-12 cm.')}
                    </p>
                </div>
                """)

                bio_html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
        <h3 style="color: #0f172a; font-size: 1.15rem; font-weight: 700; margin: 0 0 12px 0;">
            Hasil Ekstraksi Biometrik Lengkap
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
                <div style="font-size: 0.72rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">🍃 KONDISI DAUN</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-top: 5px;">{bio['kondisi_daun']}</div>
            </div>
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                <div style="font-size: 0.72rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">🎨 WARNA DAUN</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-top: 5px;">{bio['warna_daun']}</div>
            </div>
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                <div style="font-size: 0.72rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">📐 ESTIMASI LEBAR</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #047857; margin-top: 5px;">{bio.get('lebar_desc', bio['estimasi_lebar'])}</div>
            </div>
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                <div style="font-size: 0.72rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">🥣 BENTUK DAUN</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-top: 5px;">{bio['bentuk_daun']}</div>
            </div>
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                <div style="font-size: 0.72rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">🌿 JML HELAI</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-top: 5px;">{bio['jml_helai']}</div>
            </div>
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                <div style="font-size: 0.72rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">📊 LUAS KANOPI</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-top: 5px;">{bio['luas_kanopi']}</div>
            </div>
        </div>
    </div>
    """
                st.html(bio_html)
            else:
                # BUKAN PAKCOY - Dedicated Alert Card
                st.html(f"""
                <div style="background: #fef2f2; border: 1.5px solid #f87171; border-left: 6px solid #dc2626; border-radius: 12px; padding: 16px 18px; margin-bottom: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 0.78rem; font-weight: 800; color: #dc2626; text-transform: uppercase; letter-spacing: 0.05em;">
                            ⚠️ VERIFIKASI SPESIES: BUKAN TANAMAN PAKCOY
                        </span>
                        <span style="background: #dc2626; color: #ffffff; font-size: 0.72rem; font-weight: 800; padding: 2px 8px; border-radius: 6px;">
                            BUKAN PAKCOY
                        </span>
                    </div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #991b1b; margin-top: 2px; margin-bottom: 6px;">
                        {spec_name}
                    </div>
                    <div style="font-size: 0.84rem; color: #7f1d1d; line-height: 1.5; margin-bottom: 10px;">
                        Sistem mendeteksi bahwa tanaman pada foto ini adalah <strong>{spec_name}</strong> (Tingkat Keyakinan AI: <strong>{spec_conf:.1f}%</strong>). Standar lebar daun panen pakcoy (≥ 10 cm) dan jadwal panen 40 hari tidak berlaku untuk komoditas ini.
                    </div>
                    <div style="background: #ffffff; border: 1px solid #fecaca; border-radius: 8px; padding: 10px 14px; font-size: 0.82rem; color: #374151;">
                        <div style="margin-bottom: 4px;">🔍 <strong>Pembeda Utama:</strong> {spec_info.get('perbedaan_pakcoy', 'Karakter morfologi berbeda dengan pakcoy.')}</div>
                        <div>💡 <strong>Ciri Khas Terdeteksi:</strong> {spec_info.get('alasan', 'Signature daun bukan roset sendok pakcoy.')}</div>
                    </div>
                </div>
                """)

                bio_html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
        <h3 style="color: #0f172a; font-size: 1.15rem; font-weight: 700; margin: 0 0 12px 0;">
            Hasil Ekstraksi Biometrik Tanaman
        </h3>

        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 2px 8px -2px rgba(0,0,0,0.04); display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <div style="background: #fee2e2; border-radius: 10px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; border: 1.5px solid #fca5a5; color: #dc2626; font-size: 1.4rem;">
                    🌿
                </div>
                <div>
                    <div style="font-size: 0.72rem; font-weight: 800; color: #dc2626; text-transform: uppercase; letter-spacing: 0.05em;">
                        SPESIES TANAMAN TERDETEKSI
                    </div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #991b1b; margin-top: 2px;">
                        {spec_name}
                    </div>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.85rem; font-weight: 800; color: #dc2626; line-height: 1;">
                    {spec_conf:.1f}%
                </div>
                <div style="font-size: 0.74rem; font-weight: 700; color: #64748b; margin-top: 2px;">
                    Keyakinan AI
                </div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                <div style="font-size: 0.72rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">🍃 KONDISI DAUN</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-top: 5px;">{bio['kondisi_daun']}</div>
            </div>
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                <div style="font-size: 0.72rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">🎨 WARNA DAUN</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-top: 5px;">{bio['warna_daun']}</div>
            </div>
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                <div style="font-size: 0.72rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">📐 ESTIMASI LEBAR</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #dc2626; margin-top: 5px;">{bio['lebar_desc']}</div>
            </div>
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                <div style="font-size: 0.72rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">🥣 BENTUK DAUN</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-top: 5px;">{bio['bentuk_daun']}</div>
            </div>
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                <div style="font-size: 0.72rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">🌿 JML HELAI</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-top: 5px;">{bio['jml_helai']}</div>
            </div>
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                <div style="font-size: 0.72rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">📊 LUAS KANOPI</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-top: 5px;">{bio['luas_kanopi']} ({bio['luas_kanopi_pct']}%)</div>
            </div>
        </div>
    </div>
    """
                st.html(bio_html)

        # ----------------- SECTION DIAGNOSTIK AI & REKOMENDASI -----------------
        if not is_pakcoy:
            # Summary chips for non-pakcoy
            st.html(f"""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; margin-top: 14px; margin-bottom: 12px;">
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 12px; font-size: 0.8rem; color: #475569; display: flex; align-items: center; gap: 6px;">
                    <span style="font-weight: 700; color: #dc2626;">🌿 Spesies:</span> {spec_name}
                </div>
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 12px; font-size: 0.8rem; color: #475569; display: flex; align-items: center; gap: 6px;">
                    <span style="font-weight: 700; color: #0284c7;">🎯 Keyakinan AI:</span> {spec_conf:.1f}%
                </div>
                <div style="background: #fee2e2; border: 1px solid #fca5a5; border-radius: 8px; padding: 8px 12px; font-size: 0.8rem; color: #991b1b; display: flex; align-items: center; gap: 6px; font-weight: 700;">
                    ⚠️ Status Panen: BUKAN PAKCOY
                </div>
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 12px; font-size: 0.8rem; color: #475569; display: flex; align-items: center; gap: 6px;">
                    <span style="font-weight: 700; color: #64748b;">🏷️ Batch Uji:</span> {selected_plant_id}
                </div>
            </div>
            """)

            # Non-pakcoy full report card
            st.html(f"""
            <div style="background: #ffffff; border: 1.5px solid #ef4444; border-left: 6px solid #dc2626; border-radius: 14px; padding: 20px 24px; box-shadow: 0 4px 16px rgba(0,0,0,0.03); margin-top: 10px; margin-bottom: 14px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.3rem; line-height: 1;">⚠️</span>
                        <span style="font-size: 1.12rem; font-weight: 800; color: #0f172a; letter-spacing: -0.3px;">
                            Laporan Identifikasi Spesies &amp; Perbandingan Botani
                        </span>
                    </div>
                    <span style="background: #fee2e2; color: #991b1b; font-size: 0.76rem; font-weight: 700; padding: 4px 12px; border-radius: 9999px; border: 1px solid #fca5a5;">
                        Peringatan Spesies Non-Pakcoy
                    </span>
                </div>

                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin-bottom: 14px;">
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
                        <div style="font-size: 0.72rem; font-weight: 800; color: #dc2626; text-transform: uppercase; letter-spacing: 0.04em;">🌿 SPESIES TERDETEKSI</div>
                        <div style="font-size: 1.02rem; font-weight: 800; color: #0f172a; margin-top: 3px;">{spec_name}</div>
                        <div style="font-size: 0.82rem; color: #334155; margin-top: 4px;">Keyakinan AI: <strong>{spec_conf:.1f}%</strong></div>
                    </div>

                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
                        <div style="font-size: 0.72rem; font-weight: 800; color: #0284c7; text-transform: uppercase; letter-spacing: 0.04em;">🔍 CIRI MORFOLOGI</div>
                        <div style="font-size: 0.88rem; font-weight: 700; color: #0f172a; margin-top: 3px;">{spec_info.get('alasan', 'Signature daun non-pakcoy')}</div>
                        <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">{spec_info.get('karakteristik', '')}</div>
                    </div>

                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
                        <div style="font-size: 0.72rem; font-weight: 800; color: #b45309; text-transform: uppercase; letter-spacing: 0.04em;">⚠️ PERBEDAAN DENGAN PAKCOY</div>
                        <div style="font-size: 0.88rem; color: #0f172a; margin-top: 3px; line-height: 1.4;">{spec_info.get('perbedaan_pakcoy', '')}</div>
                    </div>
                </div>

                <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; padding: 12px 16px;">
                    <div style="font-size: 0.82rem; font-weight: 800; color: #92400e; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px;">
                        💡 REKOMENDASI UNTUK PENGGUNA
                    </div>
                    <div style="font-size: 0.86rem; color: #78350f; line-height: 1.5;">
                        Sistem Smart Farming ini dirancang khusus untuk memonitor siklus 40 hari tanaman <strong>Pakcoy (<em>Brassica rapa</em> L. subsp. <em>chinensis</em>)</strong>. Silakan unggah foto tanaman pakcoy untuk mengaktifkan pemantauan lebar daun (standar panen ≥ 10.0 cm), estimasi sisa hari panen, dan rekomendasi nutrisi hidroponik AB Mix.
                    </div>
                </div>
            </div>
            """)
        else:
            # Standar Pakcoy Diagnostic Card
            pred_cls = res.get("predicted_class", "Masa_Pertumbuhan")
            conf_val = round(res.get("confidence", 89.0))
            leaf_w = bio.get("leaf_width_cm", 5.0)
            canopy_cov = int(round(bio.get("luas_kanopi_pct", 32.0)))
            yellow_pct = round(bio.get("yellow_pct", 17.8), 1)
            spad_val = round(bio.get("spad_val", 41.3), 1)
            green_density = bio.get("green_density", 73)
            
            if pred_cls == "Masa_Pertumbuhan":
                leaf_w = bio.get("leaf_width_cm", 0.7)
                conf_val = round(res.get("confidence", 89.0))
                if leaf_w < 1.0:
                    fase_display = "Masa Pertumbuhan (Kotiledon / Kecambah)"
                    umur_range = "Hari ke-1 s.d. Hari ke-7 (Kecambah Dini)"
                    readiness_score = min(max(int(round((leaf_w / 10.0) * 50.0)), 4), 10)
                    norm_len = "0.8–2.0 cm"
                    norm_w = "< 1.0 cm (Fase Kotiledon)"
                    leaf_len = round(leaf_w * 1.5, 1)
                    leaf_area = round(leaf_w * leaf_len * 0.673, 1)
                    canopy_diam = round(leaf_len * 1.4, 1)
                    canopy_status_text = "yang menunjukkan helai daun masih berupa keping klorofil lembaga (kotiledon) awal perkecambahan sebelum terbentuk daun sejati"
                    nutrition_text = "perlakuan semai lembut (air murni atau larutan encer <strong>AB Mix 200–300 ppm</strong> atau <strong>EC 0,4–0,6 mS/cm</strong>) dengan penyemprotan kabut halus agar batang semai tidak patah"
                    harvest_target_text = "sehingga bibit dapat tumbuh kokoh dan siap dipindah tanam pada Hari ke-14 s.d. 18 menuju panen optimal di <strong>Hari ke-35</strong>"
                elif leaf_w <= 2.5:
                    fase_display = "Masa Pertumbuhan (Semai Daun Sejati Awal)"
                    umur_range = "Hari ke-8 s.d. Hari ke-14"
                    readiness_score = min(max(int(round(8 + (leaf_w - 1.0) * 8.0)), 8), 22)
                    norm_len = "2.0–4.8 cm"
                    norm_w = "1.0–2.5 cm (Daun Sejati Awal)"
                    leaf_len = round(leaf_w * 1.7, 1)
                    leaf_area = round(leaf_w * leaf_len * 0.673, 1)
                    canopy_diam = round(leaf_len * 1.3, 1)
                    canopy_status_text = "yang menandakan daun sejati ke-1 hingga ke-3 mulai mekar dengan tangkai semai yang mulai mengokoh"
                    nutrition_text = "nutrisi pengenalan vegetatif (larutan <strong>AB Mix kepekatan 400–600 ppm</strong> atau <strong>EC 0,8–1,2 mS/cm</strong>)"
                    harvest_target_text = "menuju fase pembesaran daun sejati dan siap panen pada <strong>Hari ke-32 s.d. Hari ke-35</strong>"
                else:
                    fase_display = "Masa Pertumbuhan (Vegetatif Aktif)"
                    umur_range = "Hari ke-15 s.d. Hari ke-22"
                    readiness_score = min(max(int(round(22 + (leaf_w - 2.5) * 8.0)), 22), 48)
                    norm_len = "5.0–11.0 cm"
                    norm_w = "2.6–5.5 cm (Vegetatif Aktif)"
                    leaf_len = round(leaf_w * 1.8, 1)
                    leaf_area = round(leaf_w * leaf_len * 0.673, 1)
                    canopy_diam = round(leaf_len * 1.25, 1)
                    canopy_status_text = "yang menunjukkan bahwa susunan roset daun masih memiliki celah antar-tangkai karena tanaman sedang aktif melebarkan helai daun muda"
                    nutrition_text = "nutrisi pertumbuhan vegetatif (larutan <strong>AB Mix kepekatan 600–900 ppm</strong> atau <strong>EC 1,2–1,8 mS/cm</strong> untuk hidroponik, maupun pupuk cair kaya Nitrogen untuk media tanah)"
                    harvest_target_text = "sehingga tanaman dapat tumbuh mulus hingga waktu panen optimal pada <strong>Hari ke-30</strong>"

                canopy_cov = int(round(bio.get("luas_kanopi_pct", 32.0)))
                yellow_pct = round(bio.get("yellow_pct", 17.8), 1)
                spad_val = round(bio.get("spad_val", 41.3), 1)
                green_density = bio.get("green_density", 73)
            elif pred_cls == "Mendekati_Panen":
                fase_display = "Mendekati Panen (Pra-Panen)"
                umur_range = "Hari ke-22 s.d. Hari ke-32"
                readiness_score = min(max(int(round(55 + (leaf_w - 6.0) * 7.5)), 55), 84)
                norm_len = "12.0–16.5 cm"
                norm_w = "7.0–9.8 cm"
                leaf_len = round(leaf_w * 1.75, 1)
                leaf_area = round(leaf_w * leaf_len * 0.673, 1)
                canopy_diam = round(leaf_len * 1.25, 1)
                canopy_status_text = "yang menunjukkan bahwa susunan roset daun mulai memadat rapat dengan tangkai pelepah yang menebal kokoh bersiap memasuki fase matang komersial"
                nutrition_text = "nutrisi penguatan tajuk dan pelepah (larutan <strong>AB Mix kepekatan 900–1.200 ppm</strong> atau <strong>EC 1,8–2,4 mS/cm</strong> untuk hidroponik) dengan rasio Kalium dan Fosfor seimbang"
                harvest_target_text = "sehingga tanaman dapat mencapai bobot panen standar dalam <strong>3–7 hari ke depan (Hari ke-35 s.d. Hari ke-38)</strong>"
            else: # Siap_Panen
                fase_display = "Siap Panen (Matang Optimal)"
                umur_range = "Hari ke-33 s.d. Hari ke-40"
                readiness_score = min(max(int(round(88 + (leaf_w - 10.0) * 3.5)), 88), 99)
                norm_len = "16.0–22.0 cm"
                norm_w = "10.0–14.5 cm"
                leaf_len = round(leaf_w * 1.7, 1)
                leaf_area = round(leaf_w * leaf_len * 0.673, 1)
                canopy_diam = round(leaf_len * 1.25, 1)
                canopy_status_text = "yang menandakan tajuk mangkok roset telah menutup penuh sempurna dengan pelepah tebal renyah siap dipanen"
                nutrition_text = "penyesuaian nutrisi akhir panen (larutan <strong>AB Mix 600–800 ppm</strong> atau pembilasan air murni selama 24 jam sebelum pemetikan)"
                harvest_target_text = "dan segera laksanakan pemanenan pagi hari ini dengan memotong tepat di pangkal akar agar kesegaran dan kerenyahan daun pakcoy tetap maksimal"

            if yellow_pct > 8.0:
                klorosis_chip = "⚠️ Perhatian : gejala klorosis daun"
                chlorosis_analysis = "Persentase warna kuning di atas 8% menandakan bahwa sebagian helai daun mengalami penurunan zat hijau daun (klorofil). Pada budidaya pakcoy, kondisi ini umumnya dipicu oleh kekurangan unsur hara makro Nitrogen (N) atau Magnesium (Mg), atau kelembapan berlebih pada media tanam."
            else:
                klorosis_chip = "✅ Status Daun: Sehat Prima (Bebas Klorosis)"
                chlorosis_analysis = "Persentase warna kuning di bawah 8% membuktikan konsentrasi klorofil prima tanpa gejala klorosis."

            if pred_cls == "Masa_Pertumbuhan":
                if leaf_w < 1.0:
                    tajuk_chip = "< 3.5 cm (Kotiledon)"
                    nutrition_brief = "Air murni / AB Mix encer 200–300 ppm"
                    harvest_target_brief = "Hari ke-30 s.d. 35 (Fase Kotiledon)"
                elif leaf_w <= 2.5:
                    tajuk_chip = "3.5–7.0 cm (Semai Awal)"
                    nutrition_brief = "AB Mix encer 400–600 ppm"
                    harvest_target_brief = "Hari ke-30 s.d. 35 (Semai Awal)"
                else:
                    tajuk_chip = "7.0–11.0 cm"
                    nutrition_brief = "AB Mix 600–900 ppm (EC 1.2–1.8 mS/cm)"
                    harvest_target_brief = "Hari ke-28 s.d. 32"
                klorosis_brief = "Penurunan klorofil ringan" if yellow_pct > 8.0 else "Klorofil optimal, fotosintesis prima"
            elif pred_cls == "Mendekati_Panen":
                tajuk_chip = "12.0–16.5 cm"
                nutrition_brief = "AB Mix 900–1.200 ppm (EC 1.8–2.4 mS/cm)"
                harvest_target_brief = "3–7 hari lagi (Hari ke-35 s.d. 38)"
                klorosis_brief = "Klorosis wajar daun tua" if yellow_pct > 8.0 else "Daun sangat prima & kokoh"
            else:
                tajuk_chip = "16.0–22.0 cm"
                nutrition_brief = "Bilas air bersih / AB Mix 600–800 ppm"
                harvest_target_brief = "Hari ke-38 s.d. 40 (Siap Panen Hari Ini!)"
                klorosis_brief = "Klorosis normal pada daun basal luar" if yellow_pct > 8.0 else "Roset tebal sempurna & segar"

            st.html(f"""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; margin-top: 14px; margin-bottom: 12px;">
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 12px; font-size: 0.8rem; color: #475569; display: flex; align-items: center; gap: 6px;">
                    <span style="font-weight: 700; color: #0f172a;">📐 Standar tajuk:</span> {tajuk_chip}
                </div>
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 12px; font-size: 0.8rem; color: #475569; display: flex; align-items: center; gap: 6px;">
                    <span style="font-weight: 700; color: #047857;">🌿 Kepekatan hijau {green_density}%</span> • Tutupan {canopy_cov}%
                </div>
                <div style="background: {'#fffbeb' if yellow_pct > 8.0 else '#f0fdf4'}; border: 1px solid {'#fde68a' if yellow_pct > 8.0 else '#bbf7d0'}; border-radius: 8px; padding: 8px 12px; font-size: 0.8rem; color: {'#b45309' if yellow_pct > 8.0 else '#166534'}; display: flex; align-items: center; gap: 6px; font-weight: 700;">
                    {klorosis_chip}
                </div>
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 12px; font-size: 0.8rem; color: #475569; display: flex; align-items: center; gap: 6px;">
                    <span style="font-weight: 700; color: #0284c7;">🏷️ Batch Tanaman:</span> {selected_plant_id}
                </div>
            </div>
            """)

            st.html(f"""
            <div style="background: #ffffff; border: 1.5px solid #10b981; border-left: 6px solid #059669; border-radius: 14px; padding: 20px 24px; box-shadow: 0 4px 16px rgba(0,0,0,0.03); margin-top: 10px; margin-bottom: 14px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 8px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.3rem; line-height: 1;">🎯</span>
                        <span style="font-size: 1.12rem; font-weight: 800; color: #0f172a; letter-spacing: -0.3px;">
                            Ringkasan Analisis AI &amp; Rekomendasi Perawatan Pakcoy
                        </span>
                    </div>
                    <span style="background: #dcfce7; color: #166534; font-size: 0.76rem; font-weight: 700; padding: 4px 12px; border-radius: 9999px; border: 1px solid #bbf7d0;">
                        Diagnostik Singkat &amp; Presisi
                    </span>
                </div>

                <!-- Grid 3 Poin Utama Biometrik & Klorofil -->
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin-bottom: 14px;">
                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
                        <div style="font-size: 0.72rem; font-weight: 800; color: #059669; text-transform: uppercase; letter-spacing: 0.04em;">🌱 FASE &amp; KESIAPAN</div>
                        <div style="font-size: 1.02rem; font-weight: 800; color: #0f172a; margin-top: 3px;">{fase_display}</div>
                        <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">{umur_range}</div>
                        <div style="font-size: 0.82rem; color: #334155; margin-top: 4px;">Keyakinan AI: <strong>{conf_val}%</strong> • Skor: <strong>{readiness_score}/100</strong></div>
                    </div>

                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
                        <div style="font-size: 0.72rem; font-weight: 800; color: #0284c7; text-transform: uppercase; letter-spacing: 0.04em;">🍃 MORFOLOGI TAJUK</div>
                        <div style="font-size: 1.02rem; font-weight: 800; color: #0f172a; margin-top: 3px;">P: ±{leaf_len} cm • L: ±{leaf_w} cm</div>
                        <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">Standar fase: {norm_w}</div>
                        <div style="font-size: 0.82rem; color: #334155; margin-top: 4px;">Tajuk: <strong>±{canopy_diam} cm</strong> • Kanopi: <strong>{canopy_cov}%</strong></div>
                    </div>

                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;">
                        <div style="font-size: 0.72rem; font-weight: 800; color: {'#b45309' if yellow_pct > 8.0 else '#166534'}; text-transform: uppercase; letter-spacing: 0.04em;">🧪 SPEKTRUM KLOROFIL</div>
                        <div style="font-size: 1.02rem; font-weight: 800; color: #0f172a; margin-top: 3px;">{spad_val} SPAD (Hijau {green_density}%)</div>
                        <div style="font-size: 0.8rem; color: {'#b45309' if yellow_pct > 8.0 else '#059669'}; margin-top: 2px;">Klorosis: <strong>{yellow_pct}%</strong></div>
                        <div style="font-size: 0.82rem; color: #334155; margin-top: 4px;">Status: <em>{klorosis_brief}</em></div>
                    </div>
                </div>

                <!-- Rekomendasi Tindakan Cepat -->
                <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 10px; padding: 12px 16px;">
                    <div style="font-size: 0.82rem; font-weight: 800; color: #166534; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px;">
                        📋 REKOMENDASI TINDAKAN CEPAT
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 8px; font-size: 0.86rem; color: #334155;">
                        <div>💧 <strong>Air:</strong> Kelembapan <strong>55%–75%</strong> (Siram 07:00 &amp; 16:00)</div>
                        <div>⚡ <strong>Nutrisi:</strong> <strong>{nutrition_brief}</strong> (pH 6.0–6.8, Suhu 20–30°C)</div>
                        <div>✂️ <strong>Tajuk:</strong> Pangkas daun kuning bawah, periksa hama ulat/kutu</div>
                        <div>🎯 <strong>Panen:</strong> Target panen optimal pada <strong>{harvest_target_brief}</strong></div>
                    </div>
                </div>
            </div>
            """)

            with st.expander("📄 Tampilkan Laporan Diagnostik AI Naratif Lengkap (Versi Detail)", expanded=False):
                st.html(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px 20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.7; color: #334155; font-size: 0.92rem;">
                    <p style="margin-bottom: 14px; text-align: justify;">
                        Berdasarkan hasil pemeriksaan visual menggunakan model kecerdasan buatan (CNN), tanaman pakcoy pada foto ini teridentifikasi berada pada fase <strong>{fase_display}</strong> (perkiraan umur tanam <strong>{umur_range}</strong>) dengan tingkat keyakinan AI sebesar <strong>{conf_val}%</strong> dan skor kesiapan panen <strong>{readiness_score}/100</strong>. Dari sisi ukuran fisik morfologi, helaian daun utama memiliki <strong>panjang daun ± {leaf_len} cm</strong> (standar normal fase ini: {norm_len}) dan <strong>lebar daun ± {leaf_w} cm</strong> (standar normal: {norm_w}), menghasilkan <strong>luas permukaan per helai sekitar {leaf_area} cm²</strong>. Bentang <strong>diameter tajuk rumpun mencapai ± {canopy_diam} cm</strong> dengan tutupan kanopi <strong>{canopy_cov}%</strong>, {canopy_status_text}.
                    </p>
                    <p style="margin-bottom: 14px; text-align: justify;">
                        Dari hasil analisis spektrum warna daun, terdeteksi gejala daun menguning (klorosis) sebesar <strong>{yellow_pct}%</strong> dengan indeks klorofil hijau <strong>{spad_val} SPAD</strong> (kepekatan warna hijau {green_density}%). {chlorosis_analysis}
                    </p>
                    <p style="margin-bottom: 0; text-align: justify;">
                        <strong>Saran Perawatan &amp; Panen:</strong> Pertahankan kelembapan ideal 55%–75% dengan jadwal penyiraman teratur. Berikan {nutrition_text}. {harvest_target_text}.
                    </p>
                </div>
                """)

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
            if is_pakcoy:
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
                
                with st.expander("🔍 Lihat Verifikasi Spesies Multi-Komoditas", expanded=False):
                    spec_prob_df = pd.DataFrame([
                        {"Komoditas": k, "Probabilitas (%)": v * 100}
                        for k, v in res.get("species_probabilities", {}).items()
                    ])
                    fig_s = px.bar(
                        spec_prob_df,
                        x="Probabilitas (%)",
                        y="Komoditas",
                        orientation="h",
                        color="Probabilitas (%)",
                        color_continuous_scale="Greens",
                        text_auto=".1f"
                    )
                    fig_s.update_layout(
                        template="plotly_white",
                        showlegend=False,
                        height=260,
                        margin=dict(l=0, r=25, t=10, b=10),
                        plot_bgcolor="rgba(248, 250, 252, 0.8)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_s, use_container_width=True)
            else:
                st.html("""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;">
                    <h4 style="margin: 0; color: #991b1b; font-size: 0.98rem; font-weight: 700;">
                        🔍 Probabilitas Klasifikasi Spesies Tanaman (CNN Multi-Class)
                    </h4>
                </div>
                """)
                spec_prob_df = pd.DataFrame([
                    {"Spesies / Tanaman": k, "Probabilitas (%)": v * 100}
                    for k, v in res.get("species_probabilities", {}).items()
                ])
                fig_spec = px.bar(
                    spec_prob_df,
                    x="Probabilitas (%)",
                    y="Spesies / Tanaman",
                    orientation="h",
                    color="Probabilitas (%)",
                    color_continuous_scale="Reds",
                    text_auto=".1f"
                )
                fig_spec.update_layout(
                    template="plotly_white",
                    showlegend=False,
                    height=300,
                    margin=dict(l=0, r=25, t=10, b=10),
                    plot_bgcolor="rgba(248, 250, 252, 0.8)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Tingkat Keyakinan AI (%)",
                    yaxis_title=None
                )
                st.plotly_chart(fig_spec, use_container_width=True)

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
                height=260 if is_pakcoy else 300,
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
                            Grad-CAM menunjukkan gradien aktivasi spasial pada area kanopi daun yang menjadi pertimbangan utama jaringan neural dalam mengenali fitur morfologi dan spesies tanaman.
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
                Bandingkan dua sampel foto tanaman (baik sesama pakcoy beda fase, maupun perbandingan pakcoy dengan tanaman selada/bayam) untuk menganalisis perkembangan kanopi daun dan klasifikasi botani.
            </p>
        </div>
        """)
        
        all_samples = glob.glob(os.path.join(sample_dir, "**", "*.jpg"), recursive=True) if os.path.isdir(sample_dir) else []
        if not all_samples and os.path.isdir(light_sample_dir):
            all_samples = glob.glob(os.path.join(light_sample_dir, "*.jpg"))
            
        sample_dict = {f"{os.path.basename(os.path.dirname(p))} - {os.path.basename(p)}": p for p in all_samples}
        sample_keys = list(sample_dict.keys())
        
        path_a = None
        path_b = None
        
        if len(sample_keys) >= 2:
            c_cmp1, c_cmp2 = st.columns(2, gap="large")
            with c_cmp1:
                st.markdown("<h4 style='color: #047857; font-size: 1rem;'>📷 Foto Sampel A (Awal / Kontrol)</h4>", unsafe_allow_html=True)
                def_idx_a = 0
                sel_a = st.selectbox("Pilih Sampel A:", sample_keys, index=def_idx_a, key="cmp_sel_a")
                path_a = sample_dict[sel_a]
                
            with c_cmp2:
                st.markdown("<h4 style='color: #0369a1; font-size: 1rem;'>📷 Foto Sampel B (Perkembangan / Uji)</h4>", unsafe_allow_html=True)
                def_idx_b = min(len(sample_keys) - 1, 1 if len(sample_keys) > 1 else 0)
                sel_b = st.selectbox("Pilih Sampel B:", sample_keys, index=def_idx_b, key="cmp_sel_b")
                path_b = sample_dict[sel_b]
        else:
            st.info("💡 Mode perbandingan membutuhkan dua foto sampel. Silakan unggah Foto A dan Foto B di bawah:")
            c_u1, c_u2 = st.columns(2)
            with c_u1:
                up_a = st.file_uploader("Unggah Foto Sampel A (JPG/PNG)", type=["jpg", "jpeg", "png"], key="cmp_up_a")
                if up_a:
                    path_a = up_a
            with c_u2:
                up_b = st.file_uploader("Unggah Foto Sampel B (JPG/PNG)", type=["jpg", "jpeg", "png"], key="cmp_up_b")
                if up_b:
                    path_b = up_b
                    
            if path_a is None or path_b is None:
                st.caption("ℹ️ Menunggu unggahan kedua foto untuk melakukan komparasi biometrik.")
                return

        with st.spinner("Memproses analisis biometrik kedua sampel foto..."):
            res_a = analyze_image_with_gradcam(path_a)
            res_b = analyze_image_with_gradcam(path_b)
            
        c_v1, c_v2 = st.columns(2, gap="large")
        with c_v1:
            st.image(res_a["original_image"], caption=f"Foto A: {res_a['predicted_label']} ({res_a['confidence']:.1f}%)", use_container_width=True)
        with c_v2:
            st.image(res_b["original_image"], caption=f"Foto B: {res_b['predicted_label']} ({res_b['confidence']:.1f}%)", use_container_width=True)

        bio_a = res_a["biometrics"]
        bio_b = res_b["biometrics"]
        
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #065f46;'>📋 Matriks Perbandingan Biometrik (Foto A vs Foto B)</h4>", unsafe_allow_html=True)
        
        comparison_matrix = pd.DataFrame([
            {"Parameter Biometrik": "Spesies Terdeteksi", "Foto A": res_a.get("species_name", "Pakcoy"), "Foto B": res_b.get("species_name", "Pakcoy"), "Analisis Perubahan": "Spesies Sama" if res_a.get("species_name") == res_b.get("species_name") else "Beda Spesies"},
            {"Parameter Biometrik": "Fase Kematangan / Status", "Foto A": bio_a["fase_label"], "Foto B": bio_b["fase_label"], "Analisis Perubahan": "Transisi Fase" if bio_a["fase_label"] != bio_b["fase_label"] else "Fase Sama"},
            {"Parameter Biometrik": "Indeks Kematangan (%)", "Foto A": f"{bio_a['maturity_pct']:.1f}%", "Foto B": f"{bio_b['maturity_pct']:.1f}%", "Analisis Perubahan": f"{bio_b['maturity_pct'] - bio_a['maturity_pct']:+.1f}%"},
            {"Parameter Biometrik": "Kondisi Daun", "Foto A": bio_a["kondisi_daun"], "Foto B": bio_b["kondisi_daun"], "Analisis Perubahan": "Stabil Segar" if bio_a["kondisi_daun"] == bio_b["kondisi_daun"] else "Perubahan Turgor"},
            {"Parameter Biometrik": "Warna Daun", "Foto A": bio_a["warna_daun"], "Foto B": bio_b["warna_daun"], "Analisis Perubahan": "Perkembangan Pigmen"},
            {"Parameter Biometrik": "Estimasi Lebar", "Foto A": bio_a["estimasi_lebar"], "Foto B": bio_b["estimasi_lebar"], "Analisis Perubahan": "Ekspansi Daun"},
            {"Parameter Biometrik": "Jumlah Helai", "Foto A": bio_a["jml_helai"], "Foto B": bio_b["jml_helai"], "Analisis Perubahan": "Pertambahan Daun"},
            {"Parameter Biometrik": "Luas Kanopi", "Foto A": f"{bio_a['luas_kanopi']} ({bio_a['luas_kanopi_pct']}%)", "Foto B": f"{bio_b['luas_kanopi']} ({bio_b['luas_kanopi_pct']}%)", "Analisis Perubahan": f"{bio_b['luas_kanopi_pct'] - bio_a['luas_kanopi_pct']:+.1f}% Area Kanopi"}
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
