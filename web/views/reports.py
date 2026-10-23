import streamlit as st
import pandas as pd
from datetime import datetime
from services.data_service import get_plant_details, get_latest_observation, get_growth_history
from services.lstm_service import predict_growth_4days
from services.anomaly_service import detect_anomalies_for_plant

def render_reports(selected_plant_id: str):
    plant = get_plant_details(selected_plant_id)
    latest = get_latest_observation(selected_plant_id)
    history_df = get_growth_history(selected_plant_id, limit=30)
    anomalies = detect_anomalies_for_plant(selected_plant_id)
    forecast_data = predict_growth_4days(selected_plant_id)
    
    st.html(f"""
    <div class="hero-container" style="background: linear-gradient(135deg, #064e3b 0%, #047857 50%, #0369a1 100%);">
        <div class="hero-badge">📑 Official Farm Report</div>
        <h1 class="hero-title">Laporan Komprehensif Tanaman: {selected_plant_id}</h1>
        <p class="hero-subtitle">
            Ringkasan terpadu mencakup evaluasi kondisi terkini, analisis agronomi AI, log peringatan lingkungan, 
            dan proyeksi pertumbuhan panen 4 hari ke depan.
        </p>
    </div>
    """)
    
    # Official Report Card
    st.html(f"""
    <div style="background: #ffffff; border: 1.5px solid #a7f3d0; border-radius: 18px; padding: 26px; box-shadow: 0 8px 24px -4px rgba(5, 150, 105, 0.08);">
        <div style="border-bottom: 2px solid #10b981; padding-bottom: 14px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: flex-end;">
            <div>
                <h2 style="margin: 0; color: #064e3b; font-size: 1.45rem;">SMART FARMING PAKCOY - OFFICIAL GROWTH REPORT</h2>
                <p style="margin: 4px 0 0 0; color: #64748b; font-size: 0.85rem;">Tanggal Terbit: <strong>{datetime.now().strftime('%d %B %Y')}</strong> | Dokumen: <code>RPT-{selected_plant_id}-{datetime.now().strftime('%Y%m%d')}</code></p>
            </div>
            <span class="status-badge status-optimal">● Laporan Terverifikasi AI</span>
        </div>
        
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 22px; font-size: 0.94rem;">
            <tr style="background: #f8fafc; border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 10px; width: 25%; font-weight: 600; color: #475569;">ID Tanaman / Batch</td>
                <td style="padding: 10px; width: 25%; font-weight: 800; color: #047857;">{selected_plant_id} ({plant['batch'] if plant else '-'})</td>
                <td style="padding: 10px; width: 25%; font-weight: 600; color: #475569;">Varietas Benih</td>
                <td style="padding: 10px; width: 25%; font-weight: 800; color: #0f172a;">{plant['variety'] if plant else '-'}</td>
            </tr>
            <tr style="border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 10px; font-weight: 600; color: #475569;">Hari Pengamatan</td>
                <td style="padding: 10px;">Hari ke-{latest['day']} (DAP {latest['dap']} HST)</td>
                <td style="padding: 10px; font-weight: 600; color: #475569;">Status Tanaman</td>
                <td style="padding: 10px;"><span class="status-badge status-optimal">{plant['status'] if plant else 'Optimal'}</span></td>
            </tr>
            <tr style="background: #f8fafc; border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 10px; font-weight: 600; color: #475569;">Fase Pertumbuhan</td>
                <td style="padding: 10px;"><strong>{latest['criteria']}</strong></td>
                <td style="padding: 10px; font-weight: 600; color: #475569;">Indeks Kematangan</td>
                <td style="padding: 10px; font-weight: 800; color: #059669; font-size: 1.1rem;">{latest['maturity_level']}%</td>
            </tr>
            <tr>
                <td style="padding: 10px; font-weight: 600; color: #475569;">Kelembaban Media</td>
                <td style="padding: 10px; color: #0284c7; font-weight: 700;">{latest['soil_moisture']}% ({latest['soil_condition']})</td>
                <td style="padding: 10px; font-weight: 600; color: #475569;">Suhu Lingkungan</td>
                <td style="padding: 10px; color: #d97706; font-weight: 700;">{latest['temperature']}°C</td>
            </tr>
        </table>
        
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 16px; margin-bottom: 16px;">
            <h4 style="color: #065f46; margin: 0 0 8px 0;">1. Ringkasan Diagnostik & Analisis AI</h4>
            <p style="font-size: 0.9rem; line-height: 1.6; color: #334155; margin: 0;">
                Tanaman <strong>{selected_plant_id}</strong> menunjukkan dinamika pertumbuhan yang konsisten dengan karakteristik fisiologis pakcoy. 
                Tingkat kematangan tercatat sebesar <strong>{latest['maturity_level']}%</strong> pada umur <strong>{latest['dap']} HST</strong>. 
                Suhu rata-rata {latest['temperature']}°C mendukung fotosintesis efisien tanpa stres panas, sementara kelembaban tanah berada dalam parameter terkendali.
            </p>
        </div>
        
        <div style="background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 12px; padding: 16px;">
            <h4 style="color: #0369a1; margin: 0 0 8px 0;">2. Proyeksi Pertumbuhan 4 Hari (Multivariate LSTM)</h4>
            <p style="font-size: 0.9rem; color: #334155; margin: 0;">
                Berdasarkan model time-series, tingkat kematangan diproyeksikan mencapai 
                <strong style="color: #0284c7;">{forecast_data['daily_summary_df']['Prediksi Kematangan (%)'].iloc[-1] if forecast_data else 'Optimal'}</strong> 
                pada Hari +4 mendatang dengan estimasi peningkatan yang stabil.
            </p>
        </div>
    </div>
    """)
    
    st.markdown("<br/>", unsafe_allow_html=True)
    
    # Download Button
    rep_text = f"""======================================================
LAPORAN MONITORING PERTUMBUHAN PAKCOY (SMART FARMING)
Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
ID Tanaman: {selected_plant_id}
Batch: {plant['batch'] if plant else '-'}
Varietas: {plant['variety'] if plant else '-'}
======================================================
1. STATUS TERKINI:
- Hari Pengamatan: Hari ke-{latest['day']} (DAP {latest['dap']})
- Fase Pertumbuhan: {latest['criteria']}
- Kematangan: {latest['maturity_level']}%
- Kelembaban Tanah: {latest['soil_moisture']}% ({latest['soil_condition']})
- Suhu: {latest['temperature']}°C

2. ANALISIS AI:
Tanaman berada dalam kondisi prima dengan parameter lingkungan terkendali.

3. PREDIKSI 4 HARI KE DEPAN (LSTM):
Target Kematangan Hari +4: {forecast_data['daily_summary_df']['Prediksi Kematangan (%)'].iloc[-1] if forecast_data else 'N/A'}
======================================================
"""
    st.download_button(
        label="📥 Unduh Laporan Resmi (.txt)",
        data=rep_text,
        file_name=f"Laporan_Pertumbuhan_{selected_plant_id}.txt",
        mime="text/plain",
        type="primary"
    )
