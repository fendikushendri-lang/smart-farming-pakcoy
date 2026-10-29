import streamlit as st
import os
import sys

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from web.styles import apply_custom_css
from database.db import init_db
from populate_db import populate
from services.data_service import get_all_plants, get_latest_observation

import importlib
import web.sensor_panels
import web.views.dashboard
import web.views.image_analysis
import web.views.prediction
import web.views.history_view
import web.views.ai_assistant
import web.views.settings
import ai.gemini_service

# Ensure views and sensor panels are always loaded fresh from disk
importlib.reload(ai.gemini_service)
importlib.reload(web.sensor_panels)
importlib.reload(web.views.dashboard)
importlib.reload(web.views.image_analysis)
importlib.reload(web.views.prediction)
importlib.reload(web.views.history_view)
importlib.reload(web.views.ai_assistant)
importlib.reload(web.views.settings)

from web.views.dashboard import render_dashboard
from web.views.prediction import render_prediction
from web.views.image_analysis import render_image_analysis
from web.views.history_view import render_history
from web.views.ai_assistant import render_ai_assistant
from web.views.settings import render_settings

# 1. Page Configuration
st.set_page_config(
    page_title="Smart Pakcoy AI - Monitoring & Irigasi Cerdas",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Vibrant & Harmonious Theme CSS
apply_custom_css()

# Ensure Database is Initialized
init_db()
populate()

# Initialize Session State
if "selected_plant_id" not in st.session_state:
    st.session_state["selected_plant_id"] = "PCK-001"

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.html("""
    <div class="sidebar-brand-card">
        <div style="font-size: 1.8rem; line-height: 1; margin-bottom: 4px;">🥬</div>
        <h2 style="color: #ffffff; margin: 0; font-size: 1.35rem; font-weight: 800; letter-spacing: -0.5px;">SMART PAKCOY AI</h2>
        <div style="font-size: 0.74rem; color: #a7f3d0; margin-top: 6px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
            <span class="pulse-dot"></span> Sistem Monitoring Aktif
        </div>
    </div>
    """)
    
    # Plant Selection Dropdown
    plants_df = get_all_plants()
    plant_options = plants_df['plant_id'].tolist() if not plants_df.empty else ["PCK-001"]
    current_idx = 0
    if st.session_state["selected_plant_id"] in plant_options:
        current_idx = plant_options.index(st.session_state["selected_plant_id"])
        
    st.markdown("<p style='font-size: 0.8rem; font-weight: 700; color: #047857; margin: 0 0 4px 2px; text-transform: uppercase;'>🎯 Pilih Tanaman / Batch:</p>", unsafe_allow_html=True)
    selected_plant = st.selectbox(
        "Fokus Tanaman Aktif:",
        plant_options,
        index=current_idx,
        label_visibility="collapsed",
        help="Memilih tanaman yang menjadi fokus analisis dashboard, timeline, dan asisten AI."
    )
    st.session_state["selected_plant_id"] = selected_plant
    
    # --- QUICK REAL-TIME PUMP STATUS BADGE ---
    latest = get_latest_observation(selected_plant)
    if latest:
        cur_moist = float(latest.get('soil_moisture', 55.0))
        cur_temp = float(latest.get('temperature', 26.5))
        dap_val = int(latest.get('dap', 1))
        
        # Check override or threshold
        if "manual_pump_override" in st.session_state:
            pump_on = st.session_state["manual_pump_override"]
        else:
            pump_on = (latest.get('pump_status') == 'ON') or (cur_moist < 50.0)
            
        pump_badge_text = "POMPA: AKTIF (MENYIRAM 💧)" if pump_on else "POMPA: STANDBY (OFF ⏸️)"
        pump_bg = "#ecfdf5" if pump_on else "#f8fafc"
        pump_border = "#10b981" if pump_on else "#cbd5e1"
        pump_text_color = "#047857" if pump_on else "#475569"
        
        st.html(f"""
        <div class="sidebar-status-box" style="background: {pump_bg}; border: 1.5px solid {pump_border};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-size: 0.72rem; color: {pump_text_color}; font-weight: 800; text-transform: uppercase;">
                    🚰 Status Pompa Real-Time
                </span>
                <span style="background: {'#d1fae5' if pump_on else '#e2e8f0'}; color: {'#065f46' if pump_on else '#334155'}; font-size: 0.68rem; font-weight: 800; padding: 2px 6px; border-radius: 4px;">
                    {'AKTIF' if pump_on else 'OFF'}
                </span>
            </div>
            <div style="font-size: 0.88rem; font-weight: 800; color: {pump_text_color}; margin-bottom: 6px;">
                {pump_badge_text}
            </div>
            <div style="display: flex; gap: 4px; margin-top: 4px; flex-wrap: wrap;">
                <span style="background: #eff6ff; border: 1px solid #bfdbfe; padding: 2px 6px; border-radius: 6px; color: #1d4ed8; font-size: 0.74rem; font-weight: 700;">
                    💧 {cur_moist:.1f}%
                </span>
                <span style="background: #fffbeb; border: 1px solid #fde68a; padding: 2px 6px; border-radius: 6px; color: #b45309; font-size: 0.74rem; font-weight: 700;">
                    🌡️ {cur_temp:.1f}°C
                </span>
                <span style="background: #f0fdf4; border: 1px solid #bbf7d0; padding: 2px 6px; border-radius: 6px; color: #047857; font-size: 0.74rem; font-weight: 700;">
                    🌱 {dap_val} HST
                </span>
            </div>
        </div>
        """)
        
    st.markdown("<p style='font-size: 0.8rem; font-weight: 700; color: #047857; margin: 12px 0 6px 2px; text-transform: uppercase;'>🧭 Menu Navigasi:</p>", unsafe_allow_html=True)
    
    # STRUKTUR MENU NAVIGASI SESUAI INSTRUKSI:
    # Dashboard Utama, Prediksi 4 Hari (LSTM), Analisis Tanaman (CNN), Histori, AI Pakcoy Assistant, Pengaturan
    menu_options = [
        "📊 Dashboard Utama",
        "📈 Prediksi 4 Hari (LSTM)",
        "📷 Analisis Tanaman (CNN)",
        "📜 Histori",
        "🤖 AI Pakcoy Assistant",
        "⚙️ Pengaturan"
    ]
    
    # Check if page was requested via session state
    default_nav_idx = 0
    if st.session_state.get("target_menu") in menu_options:
        default_nav_idx = menu_options.index(st.session_state.pop("target_menu"))
        
    menu = st.radio(
        "Pilih Halaman:",
        menu_options,
        index=default_nav_idx,
        label_visibility="collapsed"
    )
    
    st.markdown("<br/>", unsafe_allow_html=True)
    st.html("""
    <div style="background: rgba(255, 255, 255, 0.7); border: 1px solid #d1fae5; border-radius: 12px; padding: 12px; font-size: 0.73rem; color: #065f46; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
        <strong style="color: #047857;">🌱 Pakcoy Smart IoT v2.0</strong><br/>
        Otomasi Pompa + CNN Lebar Daun + LSTM<br/>
        <span style="color: #10b981;">●</span> Telemetri Aktif Berjalan
    </div>
    """)

# --- PAGE ROUTING ---
if menu == "📊 Dashboard Utama":
    render_dashboard(st.session_state["selected_plant_id"])
elif menu == "📈 Prediksi 4 Hari (LSTM)":
    render_prediction(st.session_state["selected_plant_id"])
elif menu == "📷 Analisis Tanaman (CNN)":
    render_image_analysis()
elif menu == "📜 Histori":
    render_history()
elif menu == "🤖 AI Pakcoy Assistant":
    render_ai_assistant(st.session_state["selected_plant_id"])
elif menu == "⚙️ Pengaturan":
    render_settings()
