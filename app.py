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
import web.views.monitoring
import web.views.plant_detail
import web.views.image_analysis
import web.views.prediction
import web.views.ai_assistant
import web.views.data_management
import web.views.reports
import web.views.model_performance
import web.views.settings
import ai.gemini_service

# Ensure views and sensor panels are always loaded fresh from disk
importlib.reload(ai.gemini_service)
importlib.reload(web.sensor_panels)
importlib.reload(web.views.dashboard)
importlib.reload(web.views.monitoring)
importlib.reload(web.views.plant_detail)
importlib.reload(web.views.image_analysis)
importlib.reload(web.views.prediction)
importlib.reload(web.views.ai_assistant)
importlib.reload(web.views.data_management)
importlib.reload(web.views.reports)
importlib.reload(web.views.model_performance)
importlib.reload(web.views.settings)

from web.views.dashboard import render_dashboard
from web.views.monitoring import render_monitoring
from web.views.plant_detail import render_plant_detail
from web.views.image_analysis import render_image_analysis
from web.views.prediction import render_prediction
from web.views.ai_assistant import render_ai_assistant
from web.views.data_management import render_data_management
from web.views.reports import render_reports
from web.views.model_performance import render_model_performance
from web.views.settings import render_settings

# 1. Page Configuration
st.set_page_config(
    page_title="Smart Pakcoy AI - Monitoring & Prediksi Pertumbuhan",
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
    
    # Active Plant Selector
    plants_df = get_all_plants()
    plant_options = plants_df['plant_id'].tolist()
    
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
    
    # Quick Plant Status Badge
    latest = get_latest_observation(selected_plant)
    if latest:
        status_color = "status-optimal" if latest['soil_condition'] == "Optimal" else "status-warning"
        st.html(f"""
        <div class="sidebar-status-box">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 0.72rem; color: #047857; font-weight: 800; text-transform: uppercase;">Status ({selected_plant})</span>
                <span class="status-badge {status_color}">{latest['soil_condition']}</span>
            </div>
            <div style="font-size: 0.95rem; font-weight: 800; color: #0f172a; margin-bottom: 4px;">
                {latest['criteria'].split(' (')[0]}
            </div>
            <div style="display: flex; gap: 6px; margin-top: 6px;">
                <span style="background: #f0fdf4; border: 1px solid #bbf7d0; padding: 2px 8px; border-radius: 6px; color: #047857; font-size: 0.78rem; font-weight: 700;">
                    🌱 {latest['dap']} HST
                </span>
                <span style="background: #ecfdf5; border: 1px solid #a7f3d0; padding: 2px 8px; border-radius: 6px; color: #059669; font-size: 0.78rem; font-weight: 800;">
                    📊 {latest['maturity_level']}%
                </span>
                <span style="background: #eff6ff; border: 1px solid #bfdbfe; padding: 2px 8px; border-radius: 6px; color: #1d4ed8; font-size: 0.78rem; font-weight: 700;">
                    🌡️ {latest['temperature']}°C
                </span>
            </div>
        </div>
        """)
        
    st.markdown("<p style='font-size: 0.8rem; font-weight: 700; color: #047857; margin: 12px 0 6px 2px; text-transform: uppercase;'>🧭 Menu Navigasi:</p>", unsafe_allow_html=True)
    menu_options = [
        "📊 Executive Dashboard",
        "📷 Analisis Tanaman (CNN)",
        "📋 Monitoring Tanaman",
        "🌿 Detail Tanaman",
        "📈 Prediksi 4 Hari (LSTM)",
        "🤖 AI Pakcoy Assistant",
        "💾 Manajemen Data",
        "📑 Laporan & Export",
        "⚡ Performa Model",
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
        <strong style="color: #047857;">🌱 Pakcoy AI Engine v1.0</strong><br/>
        Computer Vision + Multivariate Forecaster<br/>
        <span style="color: #10b981;">●</span> Sistem Siap Digunakan
    </div>
    """)

# --- PAGE ROUTING ---
if menu == "📊 Executive Dashboard":
    render_dashboard(st.session_state["selected_plant_id"])
elif menu == "📋 Monitoring Tanaman":
    render_monitoring()
elif menu == "🌿 Detail Tanaman":
    render_plant_detail(st.session_state["selected_plant_id"])
elif menu == "📷 Analisis Tanaman (CNN)":
    render_image_analysis()
elif menu == "📈 Prediksi 4 Hari (LSTM)":
    render_prediction(st.session_state["selected_plant_id"])
elif menu == "🤖 AI Pakcoy Assistant":
    render_ai_assistant(st.session_state["selected_plant_id"])
elif menu == "💾 Manajemen Data":
    render_data_management()
elif menu == "📑 Laporan & Export":
    render_reports(st.session_state["selected_plant_id"])
elif menu == "⚡ Performa Model":
    render_model_performance()
elif menu == "⚙️ Pengaturan":
    render_settings()
