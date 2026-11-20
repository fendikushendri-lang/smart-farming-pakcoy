import streamlit as st
import os
import markdown
from ai.gemini_service import (
    run_gemini_chat_query,
    get_gemini_api_key,
    save_gemini_api_key,
    AVAILABLE_GEMINI_MODELS
)
from services.data_service import get_plant_details, get_latest_observation

def get_current_sensor_context(selected_plant_id: str) -> dict:
    """Mengumpulkan metrik sensorik real-time dari session_state dan database."""
    latest = get_latest_observation(selected_plant_id) or {}
    p_info = get_plant_details(selected_plant_id) or {}
    
    # Ambil dari st.session_state jika ada update real-time/simulasi, fallback ke database
    moisture = st.session_state.get("active_soil_moisture", latest.get("soil_moisture", 55.6))
    temp = st.session_state.get("active_temperature", latest.get("temperature", 23.7))
    dap = st.session_state.get("active_dap", latest.get("dap", 10))
    pump = st.session_state.get("active_pump_status", latest.get("pump_status", "OFF"))
    cond = st.session_state.get("active_soil_condition", latest.get("soil_condition", "Optimal"))
    mat_lvl = latest.get("maturity_level", 18.1)
    
    # Ambil data CNN & LSTM terkini dari session_state jika ada
    cnn_data = st.session_state.get("latest_cnn_result")
    cnn_summary = "Belum ada pemindaian citra terbaru."
    if cnn_data:
        cnn_summary = f"Spesies: {cnn_data.get('species', 'Pakcoy')}, Fase: {cnn_data.get('stage', 'N/A')}, Estimasi Lebar Daun: {cnn_data.get('leaf_width', 'N/A')} cm, Status: {cnn_data.get('status', 'N/A')}"
        
    lstm_data = st.session_state.get("latest_lstm_result")
    lstm_summary = "Proyeksi pertumbuhan 4 hari dalam tren normal dan stabil."
    if lstm_data:
        lstm_summary = str(lstm_data)
        
    return {
        "plant_id": selected_plant_id,
        "name": p_info.get("name", f"Pakcoy Batch {selected_plant_id}"),
        "variety": p_info.get("variety", "Pakcoy Green Fortune"),
        "soil_moisture": float(moisture),
        "temperature": float(temp),
        "dap": int(dap),
        "pump_status": str(pump),
        "soil_condition": str(cond),
        "maturity_level": float(mat_lvl),
        "cnn_info": cnn_summary,
        "lstm_info": lstm_summary
    }

def render_ai_assistant(selected_plant_id: str):
    p_info = get_plant_details(selected_plant_id)
    p_name = p_info['name'] if p_info else selected_plant_id
    
    # 1. Ambil API Key dari .env, st.secrets, atau st.session_state
    gemini_key = get_gemini_api_key()
    gemini_model = st.session_state.get("gemini_model", "gemini-3-flash-preview")
    if gemini_model in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-lite"]:
        gemini_model = "gemini-3-flash-preview"
        st.session_state["gemini_model"] = "gemini-3-flash-preview"
    
    # 2. Inisialisasi riwayat percakapan di st.session_state.messages sesuai instruksi
    if "messages" not in st.session_state:
        # Migrasikan dari chat_messages jika ada riwayat sebelumnya
        st.session_state.messages = st.session_state.get("chat_messages", [])
    st.session_state["chat_messages"] = st.session_state.messages
    
    # 3. Kumpulkan konteks sensorik real-time
    sensor_ctx = get_current_sensor_context(selected_plant_id)
    
    # CSS Custom untuk tata letak percakapan elegan, modern, dan responsif
    st.markdown("""
    <style>
    /* Card Pembungkus Utama Chat */
    .chat-main-card {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-radius: 20px;
        padding: 24px 28px 18px 28px;
        box-shadow: 0 4px 20px -4px rgba(0, 0, 0, 0.05);
        margin-bottom: 24px;
    }
    
    /* Header Card */
    .chat-header-flex {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 14px;
        border-bottom: 1.5px solid #f1f5f9;
        margin-bottom: 16px;
    }
    .chat-header-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .chat-icon-badge {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: #ede9fe;
        color: #6366f1;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        flex-shrink: 0;
    }
    .chat-title-text {
        font-size: 1.15rem;
        font-weight: 800;
        color: #1e293b;
        line-height: 1.2;
    }
    .chat-subtitle-text {
        font-size: 0.86rem;
        color: #64748b;
        margin-top: 3px;
    }
    
    /* Sensor Context Bar */
    .sensor-context-bar {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 8px 14px;
        margin-bottom: 18px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        font-size: 0.8rem;
    }
    .sensor-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 9px;
        border-radius: 6px;
        font-weight: 700;
    }
    
    /* Chat Area */
    .chat-area-container {
        display: flex;
        flex-direction: column;
        gap: 18px;
        min-height: 380px;
        margin-bottom: 20px;
    }
    
    /* Baris Pesan Pengguna (Kanan) */
    .chat-row-user {
        display: flex;
        justify-content: flex-end;
        width: 100%;
        margin-bottom: 14px;
    }
    .chat-bubble-user-indigo {
        background: #4f46e5;
        color: #ffffff !important;
        border-radius: 22px 22px 4px 22px;
        padding: 13px 22px;
        max-width: 78%;
        font-size: 0.95rem;
        line-height: 1.55;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25);
        word-break: break-word;
    }
    .chat-bubble-user-indigo p {
        margin: 0;
        color: #ffffff !important;
    }
    
    /* Baris Pesan AI (Kiri) */
    .chat-row-ai {
        display: flex;
        justify-content: flex-start;
        align-items: flex-start;
        gap: 12px;
        width: 100%;
        margin-bottom: 14px;
    }
    .chat-avatar-ai-sprout {
        width: 38px;
        height: 38px;
        border-radius: 12px;
        background: #ede9fe;
        color: #6366f1;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        flex-shrink: 0;
        margin-top: 4px;
    }
    .chat-bubble-ai-white {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-radius: 6px 20px 20px 20px;
        padding: 16px 22px;
        color: #1e293b;
        max-width: 84%;
        font-size: 0.95rem;
        line-height: 1.65;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        word-break: break-word;
    }
    .chat-bubble-ai-white p {
        margin-bottom: 8px;
    }
    .chat-bubble-ai-white p:last-child {
        margin-bottom: 0;
    }
    .chat-bubble-ai-white strong {
        color: #0f172a;
        font-weight: 700;
    }
    
    /* Quick Prompt Buttons */
    .quick-prompt-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .quick-prompt-container {
        margin-top: 8px;
        margin-bottom: 8px;
    }
    div[data-testid="stHorizontalBlock"] .stButton button {
        border-radius: 10px;
        font-size: 0.8rem;
        padding: 4px 6px;
        font-weight: 500;
    }
    
    /* Input & Disclaimer */
    .chat-disclaimer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.78rem;
        margin-top: 10px;
        letter-spacing: 0.01em;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # --- HEADER CARD ---
    col_h1, col_h2 = st.columns([11, 1], vertical_alignment="center")
    with col_h1:
        st.html("""
        <div class="chat-header-left">
            <div class="chat-icon-badge">💬</div>
            <div>
                <div class="chat-title-text">AI Pakcoy Assistant Kamu</div>
                <div class="chat-subtitle-text">Asisten virtual cerdas terintegrasi 40 batch kebun (PCK-001 s.d. PCK-040) & tanya jawab serba bisa</div>
            </div>
        </div>
        """)
    with col_h2:
        if st.button("🗑️", help="Bersihkan riwayat percakapan", key="clear_chat_top", use_container_width=True):
            st.session_state.messages = []
            st.session_state["chat_messages"] = []
            st.rerun()

    st.markdown("<hr style='margin: 12px 0 24px 0; border: none; border-top: 1.5px solid #f1f5f9;'/>", unsafe_allow_html=True)
    
    # --- DAFTAR PESAN (CHAT MESSAGES / GELEMBUNG PERCAKAPAN) ---
    chat_box = st.container()
    with chat_box:
        # Jika riwayat kosong, tampilkan sapaan pembuka
        if not st.session_state.messages:
            welcome_text = (
                f"Halo! Selamat datang di asisten virtual **Smart Farming Pakcoy**. "
                f"Saya memantau seluruh 40 batch tanaman kebun (**PCK-001** s.d. **PCK-040**) dengan fokus aktif saat ini pada **{sensor_ctx['plant_id']}** "
                f"({sensor_ctx['dap']} HST, kelembapan tanah **{sensor_ctx['soil_moisture']:.1f}%**, suhu **{sensor_ctx['temperature']:.1f}°C**).\n\n"
                f"Ada yang bisa saya bantu terkait kondisi 40 batch kebun, panduan budidaya pakcoy, maupun pertanyaan sederhana lainnya hari ini?"
            )
            rendered_welcome = markdown.markdown(welcome_text)
            st.html(f"""
            <div class="chat-row-ai">
                <div class="chat-avatar-ai-sprout">🌱</div>
                <div class="chat-bubble-ai-white">
                    {rendered_welcome}
                </div>
            </div>
            """)
        else:
            for msg in st.session_state.messages:
                raw_content = msg.get("content", "")
                if msg["role"] == "user":
                    st.html(f"""
                    <div class="chat-row-user">
                        <div class="chat-bubble-user-indigo">
                            {raw_content}
                        </div>
                    </div>
                    """)
                else:
                    rendered_ai = markdown.markdown(raw_content)
                    st.html(f"""
                    <div class="chat-row-ai">
                        <div class="chat-avatar-ai-sprout">🌱</div>
                        <div class="chat-bubble-ai-white">
                            {rendered_ai}
                        </div>
                    </div>
                    """)
                    
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # --- 5 TOMBOL PINTASAN CEPAT (QUICK PROMPTS) ---
    st.markdown("""
    <div class="quick-prompt-label">
        <span>⚡ <strong>Pertanyaan Cepat (Quick Prompts):</strong></span>
    </div>
    """, unsafe_allow_html=True)
    
    qp1, qp2, qp3, qp4, qp5 = st.columns(5)
    clicked_prompt = None
    
    with qp1:
        if st.button("🥗 Batch Siap Panen?", key="qp_panen", use_container_width=True):
            clicked_prompt = "Batch mana saja yang sudah siap panen?"
    with qp2:
        if st.button("📋 Ringkasan 40 Batch", key="qp_ringkasan", use_container_width=True):
            clicked_prompt = "Bagaimana ringkasan kondisi 40 batch kebun saat ini?"
    with qp3:
        if st.button("🌿 Kondisi PCK-020", key="qp_pck020", use_container_width=True):
            clicked_prompt = "Bagaimana kondisi batch PCK-020 saat ini?"
    with qp4:
        if st.button("🍃 Kenapa Daun Kuning?", key="qp_kuning", use_container_width=True):
            clicked_prompt = "Kenapa daun pakcoy menguning dan bagaimana cara mengatasinya?"
    with qp5:
        if st.button("☀️ Apa Fotosintesis?", key="qp_fotosintesis", use_container_width=True):
            clicked_prompt = "Apa itu fotosintesis pada tumbuhan?"

    # --- INPUT CHAT ---
    user_input = st.chat_input("Tanyakan kondisi batch PCK-001 s.d. PCK-040, panduan budidaya, atau pertanyaan umum...")
    
    prompt_to_send = clicked_prompt or user_input
    
    if prompt_to_send:
        # Tambahkan pertanyaan user ke st.session_state.messages
        st.session_state.messages.append({"role": "user", "content": prompt_to_send})
        st.session_state["chat_messages"] = st.session_state.messages
        
        with st.spinner("AI sedang berpikir dan menganalisis respon..."):
            resp = run_gemini_chat_query(
                messages=[m for m in st.session_state.messages if m["role"] in ["user", "assistant", "model"]],
                api_key=gemini_key,
                model=gemini_model,
                sensor_context=sensor_ctx,
                current_plant_id=selected_plant_id
            )
            # Simpan respon assistant
            st.session_state.messages.append(resp)
            st.session_state["chat_messages"] = st.session_state.messages
            st.rerun()
            
    # Disclaimer footer
    st.html("""
    <div class="chat-disclaimer">
        Jawaban didukung oleh Google Gemini AI & telemetri sensor real-time. Periksa kembali sebelum mengambil keputusan penting di kebun.
    </div>
    """)
