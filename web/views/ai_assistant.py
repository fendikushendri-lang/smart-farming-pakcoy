import streamlit as st
import os
import markdown
from ai.gemini_service import run_gemini_chat_query
from services.data_service import get_plant_details

def render_ai_assistant(selected_plant_id: str):
    p_info = get_plant_details(selected_plant_id)
    p_name = p_info['name'] if p_info else selected_plant_id
    
    gemini_key = st.session_state.get("gemini_api_key", os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", "")))
    gemini_model = st.session_state.get("gemini_model", "gemini-2.0-flash")
    
    # Inisialisasi riwayat percakapan default
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
        
    # CSS Custom untuk tata letak persis seperti gambar referensi
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
        padding-bottom: 16px;
        border-bottom: 1.5px solid #f1f5f9;
        margin-bottom: 24px;
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
    col_h1, col_h2 = st.columns([10, 1])
    with col_h1:
        st.html("""
        <div class="chat-header-left">
            <div class="chat-icon-badge">💬</div>
            <div>
                <div class="chat-title-text">AI Assistant</div>
                <div class="chat-subtitle-text">Tanya jawab tentang sistem dan budidaya pakcoy</div>
            </div>
        </div>
        """)
    with col_h2:
        if st.button("🗑️", help="Bersihkan riwayat percakapan", key="clear_chat_top"):
            st.session_state["chat_messages"] = []
            st.rerun()
            
    st.markdown("<hr style='margin: 12px 0 24px 0; border: none; border-top: 1.5px solid #f1f5f9;'/>", unsafe_allow_html=True)
    
    # --- DAFTAR PESAN (CHAT MESSAGES) ---
    chat_box = st.container()
    with chat_box:
        # Jika riwayat kosong, tampilkan sapaan pembuka
        if not st.session_state["chat_messages"]:
            welcome_text = "Halo! Selamat datang di asisten virtual **Smart Farming Pakcoy**. Ada yang bisa saya bantu terkait pemantauan sensor, model AI, atau panduan budidaya tanaman pakcoy Anda hari ini?"
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
            for msg in st.session_state["chat_messages"]:
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
                    
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    
    # --- INPUT CHAT ---
    user_input = st.chat_input("Tanyakan sesuatu tentang sistem atau budidaya pakcoy...")
    
    if user_input:
        # Tambahkan pertanyaan user
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        
        with st.spinner("AI sedang memproses jawaban..."):
            resp = run_gemini_chat_query(
                messages=[m for m in st.session_state["chat_messages"] if m["role"] in ["user", "assistant"]],
                api_key=gemini_key,
                model=gemini_model,
                current_plant_id=selected_plant_id
            )
            st.session_state["chat_messages"].append(resp)
            st.rerun()
            
    # Disclaimer footer seperti pada gambar referensi
    st.html("""
    <div class="chat-disclaimer">
        Jawaban dibuat oleh AI dan bisa keliru. Periksa kembali sebelum mengambil keputusan penting.
    </div>
    """)
