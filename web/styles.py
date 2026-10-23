import streamlit as st

def apply_custom_css():
    st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #1e293b;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.02em;
    }

    /* Background tone: subtle clean fresh atmosphere */
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Top Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #064e3b 0%, #059669 45%, #10b981 90%, #34d399 100%);
        color: #ffffff;
        padding: 26px 32px;
        border-radius: 20px;
        box-shadow: 0 12px 30px -8px rgba(5, 150, 105, 0.4);
        margin-bottom: 26px;
        position: relative;
        overflow: hidden;
    }
    .hero-container::after {
        content: "🥬";
        position: absolute;
        right: 25px;
        bottom: -15px;
        font-size: 6.5rem;
        opacity: 0.18;
        transform: rotate(-12deg);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.22);
        backdrop-filter: blur(8px);
        color: #ffffff;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 5px 14px;
        border-radius: 9999px;
        border: 1px solid rgba(255, 255, 255, 0.35);
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .hero-title {
        font-size: 1.95rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        color: #ffffff;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #d1fae5;
        margin: 0;
        max-width: 85%;
        line-height: 1.5;
    }

    /* Vibrant Color KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    
    .kpi-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.1);
    }
    
    /* KPI Themes */
    .kpi-green {
        background: linear-gradient(145deg, #ffffff 60%, #ecfdf5 100%);
        border-top: 4px solid #10b981;
    }
    .kpi-blue {
        background: linear-gradient(145deg, #ffffff 60%, #f0f9ff 100%);
        border-top: 4px solid #0284c7;
    }
    .kpi-amber {
        background: linear-gradient(145deg, #ffffff 60%, #fffbeb 100%);
        border-top: 4px solid #f59e0b;
    }
    .kpi-purple {
        background: linear-gradient(145deg, #ffffff 60%, #faf5ff 100%);
        border-top: 4px solid #8b5cf6;
    }
    .kpi-emerald {
        background: linear-gradient(145deg, #ffffff 60%, #f0fdf4 100%);
        border-top: 4px solid #059669;
    }

    .kpi-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    .kpi-value-green { color: #047857; }
    .kpi-value-blue { color: #0284c7; }
    .kpi-value-amber { color: #b45309; }
    .kpi-value-purple { color: #6d28d9; }

    .kpi-delta {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 6px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 8px;
        border-radius: 6px;
    }
    .kpi-delta-up { background: #dcfce7; color: #15803d; }
    .kpi-delta-warn { background: #fef3c7; color: #b45309; }
    .kpi-delta-info { background: #e0f2fe; color: #0369a1; }
    .kpi-delta-purple { background: #f3e8ff; color: #7e22ce; }

    /* Surface & Container Cards */
    .plant-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 4px 14px -2px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .plant-card:hover {
        box-shadow: 0 10px 25px -4px rgba(16, 185, 129, 0.12);
        border-color: #a7f3d0;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .status-optimal { 
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
        color: #14532d; 
        border: 1px solid #86efac;
    }
    .status-warning { 
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
        color: #78350f; 
        border: 1px solid #fcd34d;
    }
    .status-danger { 
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
        color: #7f1d1d; 
        border: 1px solid #fca5a5;
    }

    /* Custom Streamlit Sidebar Overhaul */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #ecfdf5 45%, #f0fdf4 100%) !important;
        border-right: 1px solid #d1fae5;
    }
    
    /* Sidebar Brand Card */
    .sidebar-brand-card {
        background: linear-gradient(135deg, #065f46 0%, #059669 100%);
        padding: 18px 16px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 16px;
        box-shadow: 0 8px 20px -4px rgba(5, 150, 105, 0.35);
    }
    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #34d399;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7);
        animation: pulse 1.6s infinite;
        vertical-align: middle;
        margin-right: 6px;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }

    /* Sidebar Status Box */
    .sidebar-status-box {
        background: #ffffff;
        border: 1.5px solid #a7f3d0;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 18px;
        box-shadow: 0 4px 12px -2px rgba(16, 185, 129, 0.1);
    }

    /* Streamlit Radio / Navigation Button Styling */
    div[data-testid="stRadio"] > div {
        gap: 6px;
    }
    div[data-testid="stRadio"] label {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 8px 14px !important;
        transition: all 0.2s ease !important;
        font-weight: 600 !important;
        color: #334155 !important;
        cursor: pointer !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: #f0fdf4 !important;
        border-color: #6ee7b7 !important;
        color: #047857 !important;
        transform: translateX(3px) !important;
    }

    /* Streamlit Native Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.25) !important;
    }
    button[kind="primary"] {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        color: #ffffff !important;
    }

    /* Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.88rem;
        padding: 8px 16px;
        color: #64748b;
        background-color: transparent;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #059669 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06) !important;
    }

    /* Native Streamlit Chat Message Polish (Modern AI Style) */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        border: 1px solid #e2e8f0;
        background-color: #ffffff;
        font-size: 0.95rem;
        line-height: 1.65;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 1.5px solid #a7f3d0;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-left: 5px solid #10b981;
    }
    [data-testid="stChatMessage"] h3 {
        color: #064e3b;
        margin-top: 4px;
        margin-bottom: 10px;
        font-size: 1.15rem;
    }
    [data-testid="stChatMessage"] h4 {
        color: #047857;
        margin-top: 12px;
        margin-bottom: 6px;
        font-size: 1.02rem;
    }
    [data-testid="stChatMessage"] hr {
        margin: 12px 0;
        border: none;
        border-top: 1px solid #e2e8f0;
    }
    [data-testid="stChatMessage"] ul, [data-testid="stChatMessage"] ol {
        margin-bottom: 10px;
        padding-left: 24px;
    }
    [data-testid="stChatMessage"] li {
        margin-bottom: 4px;
    }
    [data-testid="stChatMessage"] blockquote {
        border-left: 4px solid #10b981;
        background: #f0fdf4;
        padding: 8px 14px;
        border-radius: 6px;
        margin: 10px 0;
        color: #065f46;
        font-size: 0.9rem;
    }

    /* Timeline Stepper Nodes */
    .timeline-node {
        background: #ffffff;
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(16, 185, 129, 0.1);
        transition: transform 0.2s ease;
    }
    .timeline-node:hover {
        transform: translateY(-3px);
    }
    .timeline-node-future {
        background: #f0f9ff;
        border: 2px dashed #0284c7;
        border-radius: 14px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 3px 10px rgba(2, 132, 199, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
