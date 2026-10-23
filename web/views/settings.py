import streamlit as st
import os
from database.db import get_connection
from populate_db import populate
from ai.gemini_service import test_gemini_connection, AVAILABLE_GEMINI_MODELS

def render_settings():
    st.html("""
    <div class="hero-container" style="background: linear-gradient(135deg, #064e3b 0%, #059669 45%, #2563eb 100%);">
        <div class="hero-badge">⚙️ System Configuration</div>
        <h1 class="hero-title">Pengaturan Sistem & Konfigurasi AI API</h1>
        <p class="hero-subtitle">
            Kelola integrasi Google AI Studio (Gemini), model OpenAI alternatif, serta monitor integritas database lokal.
        </p>
    </div>
    """)
    
    tab_gemini, tab_openai, tab_db = st.tabs([
        "✨ 1. Google AI Studio (Gemini)",
        "🔑 2. OpenAI API (Alternatif)",
        "🗄️ 3. Status Database & Infrastruktur"
    ])
    
    # --- TAB 1: GOOGLE AI STUDIO (GEMINI) ---
    with tab_gemini:
        st.markdown("<h3 style='color: #065f46;'>✨ Integrasi Google AI Studio (Gemini API)</h3>", unsafe_allow_html=True)
        st.markdown("""
        Google AI Studio memberikan akses langsung ke keluarga model **Gemini** generasi mutakhir yang sangat cepat,
        cerdas, dan memiliki kuota gratis untuk analisis pertanian cerdas pakcoy.
        """)
        
        current_gemini_key = st.session_state.get(
            "gemini_api_key",
            os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
        )
        
        c_key, c_model = st.columns([2.5, 1.5])
        with c_key:
            gemini_key_input = st.text_input(
                "Google AI Studio API Key (AIzaSy...)",
                value=current_gemini_key,
                type="password",
                help="Masukkan API key dari Google AI Studio. Tersimpan aman di sesi memori aplikasi."
            )
        with c_model:
            model_options = st.session_state.get("discovered_gemini_models", AVAILABLE_GEMINI_MODELS)
            current_g_model = st.session_state.get("gemini_model", "gemini-2.0-flash")
            model_idx = model_options.index(current_g_model) if current_g_model in model_options else 0
            gemini_model_choice = st.selectbox(
                "Pilihan Model Gemini",
                model_options,
                index=model_idx,
                help="Rekomendasi: gemini-2.0-flash (Generasi terbaru, super cepat, dan stabil)."
            )
            
        c_btn1, c_btn2, c_btn3 = st.columns([1.5, 1.5, 3])
        with c_btn1:
            if st.button("💾 Simpan Pengaturan Gemini", type="primary", use_container_width=True):
                st.session_state["gemini_api_key"] = gemini_key_input.strip()
                st.session_state["gemini_model"] = gemini_model_choice
                st.session_state["active_ai_provider"] = "gemini"
                os.environ["GEMINI_API_KEY"] = gemini_key_input.strip()
                os.environ["GOOGLE_API_KEY"] = gemini_key_input.strip()
                st.success(f"✅ Konfigurasi Google AI Studio ({gemini_model_choice}) berhasil disimpan!")
        with c_btn2:
            if st.button("🔌 Uji Koneksi API", use_container_width=True):
                if not gemini_key_input.strip():
                    st.warning("Silakan masukkan API Key terlebih dahulu sebelum menguji koneksi.")
                else:
                    with st.spinner("Menguji koneksi ke Google AI Studio..."):
                        is_ok, msg, models = test_gemini_connection(gemini_key_input)
                        if is_ok:
                            if models:
                                st.session_state["discovered_gemini_models"] = models
                            st.success(f"🎉 {msg}")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")
                            
        if not gemini_key_input.strip():
            st.html("""
            <div style="background: #ecfdf5; border: 1.5px solid #a7f3d0; border-radius: 12px; padding: 14px 18px; margin-top: 14px; color: #065f46; font-size: 0.88rem;">
                💡 <strong>Mode Asisten Cerdas Offline Aktif:</strong> Belum ada API key Google AI Studio yang disimpan. 
                Sistem tetap berfungsi penuh menjawab pertanyaan perkembangan tanaman, analisis kelembaban, suhu, dan deteksi anomali menggunakan engine internal berbasis data aktual 40 hari.
            </div>
            """)
            
        st.markdown("<br/>", unsafe_allow_html=True)
        # Panduan mendapatkan API key
        st.html("""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
            <div style="font-weight: 800; color: #1e293b; font-size: 0.95rem; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
                <span>📖</span> Panduan Singkat Mendapatkan API Key Google AI Studio (Gratis):
            </div>
            <ol style="margin: 0; padding-left: 20px; color: #475569; font-size: 0.88rem; line-height: 1.6;">
                <li>Buka situs resmi: <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color: #2563eb; font-weight: 700; text-decoration: underline;">https://aistudio.google.com/app/apikey</a></li>
                <li>Masuk dengan akun Google (Gmail) Anda.</li>
                <li>Klik tombol biru <strong>"Create API key"</strong> (Buat Kunci API).</li>
                <li>Pilih project Google Cloud yang ada atau buat baru dalam 1 klik.</li>
                <li>Salin kode API Key yang muncul (berawalan <code>AIzaSy...</code>).</li>
                <li>Tempelkan pada kolom input di atas, lalu klik <strong>"Simpan Pengaturan Gemini"</strong> dan <strong>"Uji Koneksi API"</strong>.</li>
            </ol>
        </div>
        """)

    # --- TAB 2: OPENAI API (ALTERNATIF) ---
    with tab_openai:
        st.markdown("<h3 style='color: #065f46;'>Integrasi OpenAI API / ChatGPT (Alternatif)</h3>", unsafe_allow_html=True)
        st.markdown("""
        Anda juga dapat menggunakan OpenAI API (GPT-4o / GPT-4o-mini) sebagai alternatif mesin LLM sekunder.
        """)
        
        current_openai_key = st.session_state.get("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))
        
        c_okey, c_omodel = st.columns([2.5, 1.5])
        with c_okey:
            openai_key_input = st.text_input(
                "OpenAI API Key (sk-...)",
                value=current_openai_key,
                type="password",
                help="Masukkan API key OpenAI Anda."
            )
        with c_omodel:
            openai_model_choice = st.selectbox(
                "Model OpenAI",
                ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                index=0,
                help="Default: gpt-4o-mini"
            )
        
        if st.button("💾 Simpan Pengaturan OpenAI", type="secondary"):
            st.session_state["openai_api_key"] = openai_key_input.strip()
            st.session_state["openai_model"] = openai_model_choice
            st.session_state["active_ai_provider"] = "openai"
            os.environ["OPENAI_API_KEY"] = openai_key_input.strip()
            st.success("Kredensial OpenAI API berhasil disimpan sebagai AI Engine aktif!")

    # --- TAB 3: STATUS DATABASE ---
    with tab_db:
        st.markdown("<h3 style='color: #065f46;'>Status Kesehatan Infrastruktur Database</h3>", unsafe_allow_html=True)
        
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM plants")
        p_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM observations")
        o_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM alerts")
        a_count = cur.fetchone()[0]
        conn.close()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.html("""
            <div class="kpi-card kpi-green">
                <div class="kpi-label">🗄️ Database Relasional</div>
                <div class="kpi-value kpi-value-green" style="font-size: 1.4rem;">SQLite Active</div>
                <div class="kpi-delta kpi-delta-up">pakcoy_smart_farming.db</div>
            </div>
            """)
        with c2:
            st.html(f"""
            <div class="kpi-card kpi-blue">
                <div class="kpi-label">📊 Rekaman Observasi</div>
                <div class="kpi-value kpi-value-blue">{o_count} Baris</div>
                <div class="kpi-delta kpi-delta-info">Data Sensorik Asli 40 Hari</div>
            </div>
            """)
        with c3:
            st.html(f"""
            <div class="kpi-card kpi-purple">
                <div class="kpi-label">🌿 Tanaman Dimonitor</div>
                <div class="kpi-value kpi-value-purple">{p_count} Batch</div>
                <div class="kpi-delta kpi-delta-purple">Multi-Plant Directory</div>
            </div>
            """)
            
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #065f46;'>🔄 Tindakan Pemeliharaan</h4>", unsafe_allow_html=True)
        if st.button("♻️ Sinkronisasi Ulang Database Dari Excel Asli"):
            with st.spinner("Menginisialisasi ulang database dari new Dataset_Pakcoy.xlsx..."):
                populate()
                st.success("Database berhasil disinkronisasi ulang dengan dataset asli!")
