# 🥬 Smart Pakcoy Monitoring & Growth Prediction System berbasis AI

Sistem cerdas pemantauan, analisis citra, dan peramalan pertumbuhan tanaman pakcoy (*Brassica rapa subsp. chinensis*) berbasis **Artificial Intelligence (CNN & Multivariate LSTM)** serta dilengkapi **AI Assistant (OpenAI ChatGPT & Tool Calling)**.

Aplikasi dibangun menggunakan **Python & Streamlit** dan menggunakan dataset asli tanpa data dummy atau parameter fiktif.

---

## 🌟 Fitur Utama

1. **Executive Dashboard:**
   - Ringkasan KPI: Total tanaman, tingkat kematangan, kondisi kelembaban, suhu lingkungan, umur tanaman (HST), dan status anomali.
   - Grafik interaktif tren kematangan (%), kelembaban tanah (%), dan suhu (°C).
   - Tabel ringkasan proyeksi 4 hari ke depan.
2. **Monitoring Tanaman:**
   - Katalog unit pemantauan (`PCK-001`, `PCK-002`, `PCK-003`, `PCK-004`, dan `PCK-MASTER`).
   - Pencarian instan dan filter status (Optimal vs Perlu Perhatian).
3. **Detail Tanaman:**
   - Timeline tahapan pertumbuhan harian dari awal tanam hingga horizon prediksi 4 hari.
   - Grafik kurva pertumbuhan aktual vs proyeksi LSTM.
   - Analisis diagnostik terstruktur & rekomendasi agronomi.
4. **Analisis Foto (CNN & Grad-CAM):**
   - Klasifikasi visual 3 fase: *Masa Pertumbuhan*, *Mendekati Panen*, dan *Siap Panen*.
   - Confidence score dan distribusi probabilitas.
   - **Explainable AI (Grad-CAM Heatmap)** yang menyorot kanopi daun yang mendasari inferensi visual model.
   - Karakteristik morfologis terdeteksi (kondisi daun, kanopi, rekomendasi).
5. **Prediksi 4 Hari (Multivariate LSTM):**
   - Peramalan multi-step 12 timestep (3 sesi/hari: 08:00, 12:00, 16:00).
   - Proyeksi simultaneous untuk Kematangan (%), Kelembaban Tanah (%), dan Suhu (°C).
   - Visualisasi sambungan data historis dan horizon masa depan.
6. **AI Pakcoy Assistant (OpenAI ChatGPT with Tool Calling):**
   - Integrasi ke OpenAI API (mendukung `gpt-4o-mini`, `gpt-4o`).
   - **Function/Tool Calling dinamis** ke database lokal SQLite (`get_plant_details`, `get_growth_history`, `get_latest_observation`, `get_growth_anomalies`, `get_plant_comparison`).
   - Otomatis mempertahankan konteks tanaman yang sedang aktif dipilih.
   - Mode cerdas offline jika API key belum diisi pengguna.
7. **Deteksi Anomali (Growth & Environment Anomaly Detection):**
   - Mendeteksi kondisi media tanam basah berlebih (*Wet* / >75%) atau kekeringan (*Dry* / <50%).
   - Mendeteksi suhu lingkungan ekstrem (>30°C).
   - Peringatan dini laju pertumbuhan stagnan pada fase kritis.
8. **Manajemen Data & Ekspor Laporan:**
   - Input data pengamatan harian baru dengan validasi otomatis.
   - Ekspor laporan pertumbuhan resmi (.txt) dan dataset observasi (.csv).
9. **Performa Model:**
   - Confusion matrix, kurva loss, accuracy, precision, recall, F1-score, MAE, dan R² score.

---

## 🏗️ Struktur Proyek

```text
smart-farming-pakcoy-new/
│
├── app.py                      # Main Streamlit Web Application
├── populate_db.py              # Inisialisasi data asli ke SQLite
├── train_models_unified.py     # Training model visual dan deret waktu
├── test_pipeline.py            # End-to-end pipeline verification
│
├── data/
│   ├── images/                 # Dataset foto bersih & deduplikasi
│   │   ├── Masa_Pertumbuhan/
│   │   ├── Mendekati_Panen/
│   │   └── Siap_Panen/
│   ├── processed/              # Metadata & timeseries_clean.csv
│   └── pakcoy_smart_farming.db # SQLite database persistence
│
├── ml/
│   ├── cnn/                    # Arsitektur & training CNN
│   │   ├── model.py
│   │   └── train.py
│   └── lstm/                   # Arsitektur & training LSTM
│       ├── model.py
│       └── train.py
│
├── models/                     # Bobot model & parameter skalar
│   ├── vision_classifier.pkl
│   ├── vision_scaler.pkl
│   ├── timeseries_regressor.pkl
│   ├── timeseries_scaler.pkl
│   ├── cnn_metrics.json
│   └── lstm_metrics.json
│
├── services/                   # Business logic layer
│   ├── data_service.py         # Query database & CRUD
│   ├── cnn_service.py          # Inferensi citra & Grad-CAM
│   ├── lstm_service.py         # Inferensi peramalan 4 hari
│   └── anomaly_service.py      # Rule-based anomaly detector
│
├── ai/
│   ├── knowledge_base.py       # Knowledge base agronomi pakcoy
│   └── openai_service.py       # OpenAI API client & Tool Calling
│
├── web/                        # Antarmuka Streamlit
│   ├── styles.py               # Custom CSS styling
│   └── views/                  # Halaman aplikasi
│       ├── dashboard.py
│       ├── monitoring.py
│       ├── plant_detail.py
│       ├── image_analysis.py
│       ├── prediction.py
│       ├── ai_assistant.py
│       ├── data_management.py
│       ├── reports.py
│       ├── model_performance.py
│       └── settings.py
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Cara Menjalankan Aplikasi

1. **Buka terminal di direktori proyek:**
   ```bash
   cd "C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new"
   ```

2. **Jalankan Streamlit:**
   ```bash
   streamlit run app.py
   ```

3. **Buka di browser:**
   Aplikasi akan terbuka otomatis di `http://localhost:8501`.

4. **Konfigurasi OpenAI API Key (Opsional):**
   - Buka menu **⚙️ Pengaturan** pada sidebar.
   - Masukkan OpenAI API Key Anda (`sk-...`) lalu klik **Simpan Pengaturan API**.
   - Jika belum memiliki API key, sistem tetap dapat beroperasi normal menggunakan *Engine Analisis Offline Cerdas*.
