import os
import re
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

# Muat variabel lingkungan dari file .env di root proyek
load_dotenv()

from services.data_service import (
    get_plant_details, get_growth_history, get_latest_observation,
    get_alerts, get_plant_comparison, get_all_plants
)
from services.anomaly_service import detect_anomalies_for_plant

# Model daftar Google AI Studio terpopuler & terverifikasi untuk chat (gemini-3-flash-preview sebagai model utama)
AVAILABLE_GEMINI_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-2.5-flash"
]

def get_gemini_api_key() -> str:
    """
    Mengambil API Key Gemini dari session_state, environment variable (.env),
    atau st.secrets secara berurutan dan aman.
    """
    try:
        import streamlit as st
        if "gemini_api_key" in st.session_state and st.session_state["gemini_api_key"]:
            return str(st.session_state["gemini_api_key"]).strip()
    except Exception:
        pass

    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                return str(st.secrets["GEMINI_API_KEY"]).strip()
            if "GOOGLE_API_KEY" in st.secrets:
                return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass

    return ""

def save_gemini_api_key(api_key: str):
    """
    Menyimpan API Key ke session_state, os.environ, dan file .env lokal
    agar dapat langsung digunakan tanpa error dan bertahan setelah reload.
    """
    clean_key = api_key.strip()
    os.environ["GEMINI_API_KEY"] = clean_key
    os.environ["GOOGLE_API_KEY"] = clean_key

    try:
        import streamlit as st
        st.session_state["gemini_api_key"] = clean_key
    except Exception:
        pass

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_file = os.path.join(project_root, ".env")
        lines = []
        if os.path.exists(env_file):
            with open(env_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

        found = False
        new_lines = []
        for line in lines:
            if line.startswith("GEMINI_API_KEY="):
                new_lines.append(f"GEMINI_API_KEY={clean_key}\n")
                found = True
            else:
                new_lines.append(line)
        if not found:
            new_lines.append(f"GEMINI_API_KEY={clean_key}\n")

        with open(env_file, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    except Exception as e:
        print(f"Peringatan: Gagal menulis ke file .env: {e}")

def is_valid_chat_model(name: str) -> bool:
    """Memfilter hanya model chat/generasi teks, membuang model TTS, audio, embedding, dll."""
    name_lower = name.lower()
    excluded = ["tts", "embedding", "imagen", "audio", "aqa", "robotics", "computer-use", "transcribe", "native-audio"]
    if any(ex in name_lower for ex in excluded):
        return False
    return "gemini" in name_lower

def get_available_models_for_key(api_key: str) -> list[str]:
    """Mengambil daftar model chat yang aktif untuk API key pengguna via google-genai SDK."""
    if not api_key or not api_key.strip():
        return AVAILABLE_GEMINI_MODELS
    try:
        client = genai.Client(api_key=api_key.strip())
        models_iter = client.models.list()
        chat_models = []
        for m in models_iter:
            m_name = m.name.replace("models/", "")
            if is_valid_chat_model(m_name):
                chat_models.append(m_name)
        if chat_models:
            return chat_models
    except Exception:
        pass
    return AVAILABLE_GEMINI_MODELS

def test_gemini_connection(api_key: str) -> tuple[bool, str, list]:
    """Menguji validitas API Key Google Gemini menggunakan SDK google-genai."""
    if not api_key or not api_key.strip():
        return False, "API Key Google Gemini belum diisi.", []
    
    clean_key = api_key.strip()
    try:
        client = genai.Client(api_key=clean_key)
        models_iter = client.models.list()
        discovered_models = []
        for m in models_iter:
            m_name = m.name.replace("models/", "")
            if is_valid_chat_model(m_name):
                discovered_models.append(m_name)
                
        if not discovered_models:
            discovered_models = AVAILABLE_GEMINI_MODELS
            
        msg = f"Berhasil terhubung ke Google Gemini API! ({len(discovered_models)} model aktif ditemukan)"
        return True, msg, discovered_models
    except ClientError as e:
        return False, f"Autentikasi gagal: API Key tidak valid ({e.message if hasattr(e, 'message') else str(e)})", []
    except Exception as e:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={clean_key}"
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                return True, "Berhasil terhubung ke Google Gemini API!", AVAILABLE_GEMINI_MODELS
            else:
                err_data = res.json() if res.headers.get("content-type") == "application/json" else {}
                err_msg = err_data.get("error", {}).get("message", res.text)
                return False, f"Gagal autentikasi ke Google Gemini (HTTP {res.status_code}): {err_msg}", []
        except Exception:
            pass
        return False, f"Terjadi kesalahan saat menguji koneksi: {str(e)}", []

def extract_batch_id_from_text(text: str) -> str:
    """Mengekstrak ID batch spesifik dari teks pertanyaan pengguna (misal 'PCK-015', 'batch 20')."""
    if not text:
        return None
    # Cek format PCK-xxx atau PCK xxx
    m = re.search(r'pck[-_\s]?(\d{1,3})', text.lower())
    if m:
        num = int(m.group(1))
        if 1 <= num <= 40:
            return f"PCK-{num:03d}"
    # Cek format batch xx
    m2 = re.search(r'batch\s+(\d{1,2})', text.lower())
    if m2:
        num = int(m2.group(1))
        if 1 <= num <= 40:
            return f"PCK-{num:03d}"
    return None

def get_greenhouse_summary_context() -> str:
    """Mengumpulkan ringkasan status 40 batch kebun (PCK-001 s.d. PCK-040) dari database."""
    try:
        plants_df = get_all_plants()
        if plants_df.empty:
            return "Informasi inventaris batch sedang diperbarui."
        
        valid_plants = plants_df[plants_df['plant_id'].str.match(r'^PCK-\d{3}$')]
        total_batches = len(valid_plants)
        harvest_ready = valid_plants[valid_plants['current_dap'] >= 35]['plant_id'].tolist()
        pre_harvest = valid_plants[(valid_plants['current_dap'] >= 30) & (valid_plants['current_dap'] < 35)]['plant_id'].tolist()
        vegetative = valid_plants[(valid_plants['current_dap'] >= 15) & (valid_plants['current_dap'] < 30)]['plant_id'].tolist()
        seedling = valid_plants[(valid_plants['current_dap'] >= 5) & (valid_plants['current_dap'] < 15)]['plant_id'].tolist()
        semai = valid_plants[valid_plants['current_dap'] < 5]['plant_id'].tolist()
        
        summary = f"""INVENTARIS KEBUN LENGKAP ({total_batches} BATCH AKTIF PCK-001 s.d. PCK-040):
- Total Batch: {total_batches} Batch (PCK-001 s.d. PCK-040)
- Fase Semai (1–4 HST): {len(semai)} Batch ({semai[0] if semai else 'PCK-001'} s.d. {semai[-1] if semai else 'PCK-004'})
- Fase Pembibitan (5–14 HST): {len(seedling)} Batch ({seedling[0] if seedling else 'PCK-005'} s.d. {seedling[-1] if seedling else 'PCK-014'})
- Fase Vegetatif Aktif (15–29 HST): {len(vegetative)} Batch ({vegetative[0] if vegetative else 'PCK-015'} s.d. {vegetative[-1] if vegetative else 'PCK-029'})
- Fase Pra-Panen (30–34 HST): {len(pre_harvest)} Batch ({pre_harvest[0] if pre_harvest else 'PCK-030'} s.d. {pre_harvest[-1] if pre_harvest else 'PCK-034'})
- Fase Siap Panen Prima & Batas Panen (35–40 HST): {len(harvest_ready)} Batch ({', '.join(harvest_ready)})"""
        return summary
    except Exception as e:
        return "Inventaris: 40 Batch lengkap dari PCK-001 (1 HST) hingga PCK-040 (40 HST)."

def build_system_prompt(sensor_context: dict = None, current_plant_id: str = "PCK-001", user_query: str = "") -> str:
    """
    Membangun system prompt fleksibel, cerdas, dan informatif untuk model Gemini.
    Mencakup data sensor real-time batch aktif, inventaris seluruh 40 batch (PCK-001 s.d. PCK-040),
    serta aturan menjawab SEMUA pertanyaan sederhana secara bersahabat dan jelas.
    """
    ctx = sensor_context or {}
    batch_id = ctx.get("plant_id") or current_plant_id or "PCK-001"
    
    latest = get_latest_observation(batch_id) or {}
    p_info = get_plant_details(batch_id) or {}
    
    plant_name = ctx.get("name") or p_info.get("name", f"Pakcoy Batch {batch_id}")
    dap = ctx.get("dap", latest.get("dap", 1))
    moisture = ctx.get("soil_moisture", latest.get("soil_moisture", 55.6))
    temp = ctx.get("temperature", latest.get("temperature", 23.7))
    pump = ctx.get("pump_status", latest.get("pump_status", "OFF"))
    condition = ctx.get("soil_condition", latest.get("soil_condition", "Optimal"))
    maturity_lvl = ctx.get("maturity_level", latest.get("maturity_level", 18.1))
    leaf_w = ctx.get("leaf_width", latest.get("leaf_width", 3.0))
    
    cnn_info = ctx.get("cnn_info", "Belum ada analisis foto baru.")
    lstm_info = ctx.get("lstm_info", "Proyeksi pertumbuhan 4 hari dalam tren optimal dan stabil.")
    inventory_summary = get_greenhouse_summary_context()

    # Cek apakah pengguna menanyakan batch spesifik lain di luar batch_id saat ini
    queried_batch_info = ""
    target_q_bid = extract_batch_id_from_text(user_query)
    if target_q_bid and target_q_bid != batch_id:
        q_latest = get_latest_observation(target_q_bid) or {}
        q_pinfo = get_plant_details(target_q_bid) or {}
        if q_latest:
            queried_batch_info = f"""
KONTEKS TAMBAHAN BATCH YANG DITANYAKAN PENGGUNA ({target_q_bid}):
- Nama: {q_pinfo.get('name', target_q_bid)}
- Umur: {q_latest.get('dap', 1)} HST
- Kelembapan Tanah: {q_latest.get('soil_moisture', 55.0):.1f}% ({q_latest.get('soil_condition', 'Optimal')})
- Suhu Lingkungan: {q_latest.get('temperature', 24.0):.1f}°C
- Status Pompa: {q_latest.get('pump_status', 'OFF')}
- Lebar Daun: {q_latest.get('leaf_width', 5.0):.1f} cm | Kematangan: {q_latest.get('maturity_level', 50.0):.1f}%
- Status Panen: {'Siap Panen' if q_latest.get('dap', 1) >= 35 else 'Masa Pertumbuhan Optimal'}"""

    prompt = f"""Anda adalah asisten virtual pintar untuk aplikasi **Smart Farming Pakcoy** yang ramah, membantu, komunikatif, dan cerdas.

{inventory_summary}

KONTEKS SENSOR BATCH FOKUS AKTIF ({batch_id}):
- ID Batch / Tanaman: {batch_id} ({plant_name})
- Umur Tanaman: {dap} HST (Hari Setelah Tanam) [Siklus budidaya: 1–40 HST, masa panen prima: 35–38 HST, batas maksimal: 40 HST]
- Kelembapan Tanah: {moisture:.1f}% (Kondisi: {condition}, rentang ideal: 50% - 65%)
- Suhu Lingkungan: {temp:.1f}°C (Rentang ideal: 22°C - 28°C)
- Status Pompa Irigasi: {pump} (Otomatis ON menyiram jika kelembapan < 50%, OFF standby jika >= 50%)
- Lebar Daun Terkini: {leaf_w:.1f} cm | Tingkat Kematangan: {maturity_lvl:.1f}%
- Hasil Analisis Visual (CNN): {cnn_info}
- Proyeksi Pertumbuhan (LSTM): {lstm_info}
{queried_batch_info}

PANDUAN PERILAKU & ATURAN MENJAWAB:
1. **Dapat Menjawab SEMUA Pertanyaan Sederhana:**
   - Anda dirancang untuk menjawab **semua pertanyaan sederhana** yang diajukan pengguna dengan senang hati, jelas, ringkas, dan mudah dipahami.
   - Jangan pernah menolak pertanyaan sederhana hanya karena berada di luar topik pertanian. Jawab pertanyaan apa pun (seperti hitungan matematika, definisi umum, biologi, sains dasar, cuaca, teknologi, atau sapaan sehari-hari).

2. **Pertanyaan Seputar Pakcoy & 40 Batch Kebun (PCK-001 s.d. PCK-040):**
   - Jawab secara akurat, praktis, dan ilmiah.
   - Anda menguasai data seluruh 40 batch (`PCK-001` hingga `PCK-040`).
   - Jika pengguna menanyakan kondisi batch tertentu (misal: "bagaimana kondisi PCK-020?"), jelaskan umur HST, kelembapan tanah, suhu, lebar daun, dan status panen batch tersebut berdasarkan data di atas.
   - Jika pengguna menanyakan "batch mana yang siap dipanen", sebutkan bahwa batch PCK-035 hingga PCK-040 telah siap panen (umur >= 35 HST, daun lebar >= 10 cm, kematangan >= 89%).

3. **Format Bahasa:**
   - Bahasa Indonesia yang santun, ramah, mengalir alami, dan komunikatif.
   - Berikan penekanan teks tebal (**bold**) pada kata kunci penting dan angka.
"""
    return prompt

def generate_offline_fallback(messages: list, sensor_context: dict = None, current_plant_id: str = "PCK-001") -> str:
    """
    Engine offline cerdas dan komprehensif yang mampu menjawab SEMUA pertanyaan sederhana:
    - Informasi seluruh 40 batch kebun (PCK-001 s.d. PCK-040)
    - Kueri spesifik batch mana pun
    - Panduan agronomis budidaya pakcoy lengkap (hama, penyakit, nutrisi, jarak tanam, dll.)
    - Perhitungan matematika
    - Pengetahuan umum dasar
    - Sapaan & obrolan santai
    """
    ctx = sensor_context or {}
    batch_id = ctx.get("plant_id") or current_plant_id or "PCK-001"
    latest = get_latest_observation(batch_id) or {}
    
    dap = ctx.get("dap", latest.get("dap", 1))
    moisture = ctx.get("soil_moisture", latest.get("soil_moisture", 55.6))
    temp = ctx.get("temperature", latest.get("temperature", 23.7))
    pump = ctx.get("pump_status", latest.get("pump_status", "OFF"))
    condition = ctx.get("soil_condition", latest.get("soil_condition", "Optimal"))
    mat_val = ctx.get("maturity_level", latest.get("maturity_level", 18.1))
    leaf_w = ctx.get("leaf_width", latest.get("leaf_width", 3.0))

    last_user_msg = messages[-1]["content"] if messages else ""
    msg_lower = last_user_msg.lower().strip()
    
    # Waktu sapaan
    now_hour = datetime.now().hour
    time_greeting = "pagi" if 5 <= now_hour < 11 else ("siang" if 11 <= now_hour < 15 else ("sore" if 15 <= now_hour < 18 else "malam"))
    if "pagi" in msg_lower:
        time_greeting = "pagi"
    elif "siang" in msg_lower:
        time_greeting = "siang"
    elif "sore" in msg_lower:
        time_greeting = "sore"
    elif "malam" in msg_lower:
        time_greeting = "malam"

    # =========================================================================
    # 1. CEK KUERI BATCH SPESIFIK (PCK-001 s.d. PCK-040)
    # =========================================================================
    target_bid = extract_batch_id_from_text(msg_lower)
    if target_bid:
        p_info = get_plant_details(target_bid)
        p_latest = get_latest_observation(target_bid)
        if p_info and p_latest:
            b_dap = p_latest.get('dap', p_info.get('current_dap', 1))
            b_moist = p_latest.get('soil_moisture', 55.0)
            b_temp = p_latest.get('temperature', 24.0)
            b_pump = p_latest.get('pump_status', 'OFF')
            b_cond = p_latest.get('soil_condition', 'Optimal')
            b_leaf = p_latest.get('leaf_width', 5.0)
            b_mat = p_latest.get('maturity_level', 50.0)
            
            b_stage = "Semai (Perkecambahan)" if b_dap <= 4 else (
                "Pembibitan (Seedling)" if b_dap <= 14 else (
                    "Vegetatif Aktif" if b_dap <= 29 else (
                        "Pra-Panen (Roset Daun)" if b_dap <= 34 else (
                            "Siap Panen Prima" if b_dap <= 39 else "Batas Panen Maksimal"
                        )
                    )
                )
            )
            pump_desc = "sedang **AKTIF (Menyiram)**" if b_pump == "ON" else "dalam posisi **STANDBY (OFF)**"
            harvest_note = f"Tanaman telah mencapai kesiapan panen dengan lebar daun **{b_leaf:.1f} cm**." if b_dap >= 35 else f"Estimasi panen sekitar **{40 - b_dap} hari lagi** (pada 35–40 HST)."

            return (
                f"Berikut data telemetri *real-time* untuk batch **{target_bid}**:\n\n"
                f"- **Nama Batch:** {p_info.get('name')}\n"
                f"- **Fase & Umur:** **{b_stage}** — **{b_dap} HST**\n"
                f"- **Kelembapan Tanah:** **{b_moist:.1f}%** ({b_cond}, ideal: 50%–65%)\n"
                f"- **Suhu Lingkungan:** **{b_temp:.1f}°C** (ideal: 22°C–28°C)\n"
                f"- **Status Pompa Irigasi:** {pump_desc}\n"
                f"- **Morfologi Daun:** Lebar daun **{b_leaf:.1f} cm** | Tingkat Kematangan **{b_mat:.1f}%**\n\n"
                f"{harvest_note}"
            )

    # =========================================================================
    # 2. PERTANYAAN INVENTARIS SELURUH BATCH (PCK-001 s.d. PCK-040)
    # =========================================================================
    if any(k in msg_lower for k in ["semua batch", "daftar batch", "ada berapa batch", "total batch", "pck-001 sampai pck-040", "pck-001 s.d. pck-040", "semua tanaman", "ringkasan 40 batch", "ringkasan batch", "kondisi 40 batch"]):
        return (
            "Sistem **Smart Farming Pakcoy** saat ini mengelola tepat **40 Batch Tanaman Aktif** (**PCK-001** s.d. **PCK-040**) dengan rincian tahapan budidaya sebagai berikut:\n\n"
            "1. **Fase Semai (1–4 HST):** Batch **PCK-001** s.d. **PCK-004** (Kecambah & kotiledon muncul).\n"
            "2. **Fase Pembibitan (5–14 HST):** Batch **PCK-005** s.d. **PCK-014** (Daun sejati 2–4 helai, irigasi presisi).\n"
            "3. **Fase Vegetatif Aktif (15–29 HST):** Batch **PCK-015** s.d. **PCK-029** (Pertumbuhan roset dan lebar daun pesat).\n"
            "4. **Fase Pra-Panen (30–34 HST):** Batch **PCK-030** s.d. **PCK-034** (Tajuk rimbun, mendekati jendela panen).\n"
            "5. **Fase Siap Panen Prima (35–39 HST):** Batch **PCK-035** s.d. **PCK-039** (Lebar daun >= 10 cm, kematangan >= 89%, sangat prima untuk dipetik).\n"
            "6. **Batas Panen Maksimal (40 HST):** Batch **PCK-040** (Batas akhir siklus panen komersial agar serat daun tidak mengeras).\n\n"
            "Anda dapat memilih batch mana saja pada menu dropdown di sidebar untuk memantau grafik sensor dan analisis AI-nya secara mendetail!"
        )

    # Pertanyaan: Batch mana yang siap panen?
    if any(k in msg_lower for k in ["mana yang siap panen", "mana yang bisa dipanen", "siap panen", "siap dipanen", "harvest ready"]):
        return (
            "Berdasarkan standar umur dan biometrik lebar daun (>= 10 cm serta kematangan >= 89%), batch yang saat ini **Siap Dipanen** adalah:\n\n"
            "- **PCK-035** (35 HST): Lebar daun ~10.1 cm (Mulai masuk jendela panen komersial prima)\n"
            "- **PCK-036** (36 HST): Lebar daun ~10.5 cm (Kualitas panen prima)\n"
            "- **PCK-037** (37 HST): Lebar daun ~10.8 cm (Kualitas panen prima)\n"
            "- **PCK-038** (38 HST): Lebar daun ~11.0 cm (Puncak bobot panen prima)\n"
            "- **PCK-039** (39 HST): Lebar daun ~11.1 cm (Sangat disarankan panen hari ini)\n"
            "- **PCK-040** (40 HST): Lebar daun ~11.2 cm (**Batas Maksimal Panen** — Wajib segera dipanen agar pelepah tidak alot).\n\n"
            "Total ada **6 batch** yang siap dipanen saat ini!"
        )

    # Pertanyaan: Batch mana yang masih semai atau bibit?
    if any(k in msg_lower for k in ["mana yang semai", "batch semai", "batch bibit", "persemaian"]):
        return (
            "Batch yang saat ini berada dalam tahap pembibitan dan persemaian adalah:\n\n"
            "- **Fase Semai (1–4 HST):** **PCK-001** s.d. **PCK-004** (Benih baru berkecambah, membutuhkan media lembap dan sinar matahari pagi).\n"
            "- **Fase Pembibitan / Seedling (5–14 HST):** **PCK-005** s.d. **PCK-014** (Sudah memiliki 2–4 helai daun sejati, bersiap masuk fase vegetatif)."
        )

    # =========================================================================
    # 3. KALKULATOR MATEMATIKA SEDERHANA
    # =========================================================================
    # Deteksi pola operasi hitungan (misal: "berapa 25 x 4?", "10 + 20", "100 / 5", dll.)
    math_pattern = re.search(r'(\d+[\.,]?\d*)\s*([\+\-\*\/xX]|kali|bagi|tambah|kurang)\s*(\d+[\.,]?\d*)', msg_lower)
    if math_pattern or any(w in msg_lower for w in ["hitung", "berapa hasil", "tambah", "kurang", "kali", "bagi", "x", "+", "-", "/"]):
        try:
            expr_str = msg_lower.replace("x", "*").replace("X", "*").replace("kali", "*").replace("bagi", "/").replace("tambah", "+").replace("kurang", "-")
            found = re.search(r'([\d\.\s\+\-\*\/]+)', expr_str)
            if found:
                cand = found.group(1).strip()
                if any(op in cand for op in ["+", "-", "*", "/"]):
                    calc_res = eval(cand, {"__builtins__": {}}, {})
                    if isinstance(calc_res, float):
                        calc_res = round(calc_res, 3)
                    return f"Hasil dari perhitungan **{cand.replace('*', 'x')}** adalah **{calc_res}**."
        except Exception:
            pass

    # =========================================================================
    # 4. PENGETAHUAN & PANDUAN AGRONOMIS PAKCOY LENGKAP
    # =========================================================================
    # Apa itu Pakcoy?
    if any(k in msg_lower for k in ["apa itu pakcoy", "tentang pakcoy", "sawi sendok", "nama latin pakcoy"]):
        return (
            "**Pakcoy** (*Brassica rapa* subsp. *chinensis*) atau yang sering disebut **sawi sendok** adalah jenis sayuran daun hijau dari keluarga kubis-kubisan (*Brassicaceae*). "
            "Tanaman ini memiliki pelepah daun yang tebal, putih atau kehijauan berbentuk sendok, serta helaian daun yang renyah dan kaya nutrisi seperti vitamin A, C, K, asam folat, serta kalsium. "
            "Pakcoy sangat digemari karena siklus tanamnya yang singkat (**30–40 hari** setelah tanam) dan mudah dibudidayakan di polybag maupun hidroponik."
        )

    # Mengapa daun menguning?
    if any(k in msg_lower for k in ["menguning", "daun kuning", "kenapa kuning", "penyebab kuning"]):
        return (
            "Daun pakcoy yang menguning umumnya disebabkan oleh beberapa faktor:\n\n"
            "1. **Kekurangan Nitrogen (N):** Gejalanya daun bagian bawah menguning merata terlebih dahulu. Berikan pupuk berkadar N tinggi seperti NPK atau pupuk organik.\n"
            "2. **Kelebihan Air (*Overwatering*):** Media tanam yang terlalu basah (> 70%) membuat akar kekurangan oksigen dan membusuk, sehingga daun layu dan menguning.\n"
            "3. **Kekurangan Sinar Matahari:** Pakcoy membutuhkan minimal 5–7 jam sinar matahari per hari agar klorofil daun tetap hijau segar.\n"
            "4. **Serangan Hama/Penyakit:** Misalnya jamur bercak daun atau busuk basah."
        )

    # Mengapa daun berlubang / hama?
    if any(k in msg_lower for k in ["berlubang", "bolong", "daun bolong", "hama", "ulat", "kutu daun"]):
        return (
            "Daun pakcoy yang berlubang biasanya merupakan tanda serangan **hama daun**, terutama:\n\n"
            "1. **Ulat Grayak (*Spodoptera litura*) / Ulat Tritip:** Memakan helaian daun dari bawah hingga tersisa seratnya.\n"
            "2. **Belalang atau Kepik Hijau:** Membuat lubang-lubang acak pada daun.\n"
            "3. **Kutu Daun (*Aphids*):** Menghisap cairan daun hingga keriting dan menguning.\n\n"
            "**Solusi Pengendalian Alami:**\n"
            "- Lakukan pemeriksaan visual rutin pada pagi dan sore hari (ambil ulat secara manual jika sedikit).\n"
            "- Semprotkan **pestisida nabati** seperti larutan air rebusan daun mimba atau ekstrak bawang putih + sedikit sabun cair lembut (1–2 kali seminggu)."
        )

    # Jarak Tanam
    if any(k in msg_lower for k in ["jarak tanam", "jarak tanam pakcoy", "spasi tanam"]):
        return (
            "Jarak tanam ideal untuk pakcoy adalah **15 cm x 15 cm** atau **20 cm x 20 cm**. "
            "Jarak ini memberikan ruang yang cukup bagi tajuk roset daun untuk melebar hingga lebih dari 10 cm, "
            "memaksimalkan penangkapan sinar matahari, dan menjaga sirkulasi udara antar-tanaman agar tidak lembap berlebih yang memicu jamur."
        )

    # Kapan panen / Berapa lama panen?
    if any(k in msg_lower for k in ["kapan panen", "berapa lama panen", "umur panen", "masa panen", "waktu panen"]):
        return (
            "Tanaman pakcoy umumnya siap dipanen pada umur **30 hingga 40 HST** (Hari Setelah Tanam):\n\n"
            "- **Jendela Panen Prima (35–38 HST):** Pada fase ini bobot tanaman optimal, daun hijau segar lebar (>= 10 cm), pelepah renyah manis, dan kematangan mencapai >= 89%.\n"
            "- **Batas Maksimal (40 HST):** Tanaman wajib dipanen paling lambat pada 40 HST untuk mencegah daun mengeras berserat dan munculnya bunga (*bolting*)."
        )

    # Pupuk & Nutrisi
    if any(k in msg_lower for k in ["pupuk", "nutrisi", "npk", "ab mix", "ppm", "dosis pupuk"]):
        return (
            "Rekomendasi pemupukan dan nutrisi untuk pakcoy:\n\n"
            "1. **Media Tanah / Polybag:** Gunakan pupuk dasar kompos/kandang matang, dilanjutkan pemupukan susulan **NPK 16-16-16** seminggu sekali (dosis 2–3 gram per liter air).\n"
            "2. **Sistem Hidroponik:** Gunakan larutan **AB Mix Sayuran Daun** dengan kepekatan bertahap:\n"
            "   - Semai (1–10 HST): 400–600 PPM\n"
            "   - Vegetatif (11–25 HST): 800–1000 PPM\n"
            "   - Pra-Panen (26–40 HST): 1050–1400 PPM (EC 1.5–2.0, pH 6.0–6.8)."
        )

    # Etiolasi / Kutilang
    if any(k in msg_lower for k in ["etiolasi", "kutilang", "batang kurus", "batang tinggi"]):
        return (
            "**Etiolasi** (populer disebut bibit **kutilang**: *kurus, tinggi, langsing*) terjadi ketika bibit kekurangan sinar matahari sehingga batangnya tumbuh memanjang mencari arah cahaya dengan batang yang rapuh. "
            "**Pencegahan:** Begitu benih berkecambah dan daun lembaga (kotiledon) mulai muncul (umur 2–3 HST), segera jemur bibit di tempat yang terkena sinar matahari pagi penuh."
        )

    # Media Tanam
    if any(k in msg_lower for k in ["media tanam", "tanah apa", "campuran media"]):
        return (
            "Media tanam pakcoy yang ideal harus gembur, subur, dan memiliki aerasi (drainase) yang baik. "
            "Campuran yang direkomendasikan adalah perbandingan **2 : 1 : 1** antara **tanah humus gembur**, **arang sekam (atau sekam bakar)**, dan **pupuk kompos/kandang yang sudah matang**."
        )

    # Cara Panen
    if any(k in msg_lower for k in ["cara panen", "teknik panen", "memanen"]):
        return (
            "Ada dua cara memanen pakcoy:\n\n"
            "1. **Memotong Pangkal Batang:** Potong pangkal batang sekitar 1 cm di atas media tanam menggunakan pisau tajam bersih. Cara ini sangat rapi dan bebas kotoran tanah.\n"
            "2. **Mencabut Bersama Akar:** Cabut perlahan tanaman bersama akar-akarnya, lalu cuci bersih akarnya dengan air mengalir. Cara ini menjaga kesegaran pakcoy lebih tahan lama saat disimpan."
        )

    # Kelembaban & Suhu Optimal
    if any(k in msg_lower for k in ["kelembaban ideal", "kelembapan ideal", "suhu ideal", "suhu pakcoy"]):
        return (
            "Kondisi lingkungan ideal untuk pertumbuhan optimal pakcoy adalah:\n\n"
            "- **Kelembaban Tanah:** **50% hingga 65%** (Pompa otomatis akan menyala jika kelembaban turun di bawah 50%).\n"
            "- **Suhu Lingkungan Greenhouse:** **22°C hingga 28°C**.\n"
            "- **Pencahayaan:** Minimal 5–8 jam sinar matahari per hari."
        )

    # =========================================================================
    # 5. PENGETAHUAN UMUM SEDERHANA & SAINS
    # =========================================================================
    if "fotosintesis" in msg_lower:
        return (
            "**Fotosintesis** adalah proses biologis di mana tumbuhan berklorofil (zat hijau daun) membuat zat makanan berupa glukosa dengan menyerap air (H2O) dari akar dan karbon dioksida (CO2) dari udara, dibantu oleh energi sinar matahari. "
            "Reaksi ini juga menghasilkan gas oksigen (O2) yang sangat penting bagi kehidupan di bumi."
        )

    if any(k in msg_lower for k in ["klorofil", "zat hijau daun"]):
        return (
            "**Klorofil** (sering disebut sebagai **zat hijau daun**) adalah pigmen warna hijau alami yang terdapat pada kloroplas sel tumbuhan, alga, dan beberapa jenis bakteri fotosintetik.\n\n"
            "**Fungsi Utama Klorofil:**\n"
            "1. **Menyerap Energi Sinar Matahari:** Terutama spektrum cahaya biru dan merah, lalu memantulkan spektrum hijau (sehingga daun pakcoy tampak berwarna hijau segar).\n"
            "2. **Pusat Reaksi Fotosintesis:** Mengubah energi radiasi cahaya matahari menjadi energi kimia (glukosa) yang menjadi sumber nutrisi utama bagi pertumbuhan tanaman.\n"
            "3. **Indikator Kesehatan Tanaman:** Kadar klorofil yang tinggi menandakan pasokan nitrogen cukup dan tanaman sehat prima. Jika kekurangan nitrogen atau air, klorofil akan terurai dan daun menguning (*klorosis*)."
        )

    if any(k in msg_lower for k in ["stomata", "pori daun"]):
        return (
            "**Stomata** adalah bukaan atau celah mikroskopis pada epidermis daun (kebanyakan di permukaan bawah daun) yang diapit oleh sepasang sel penjaga (*guard cells*). "
            "Fungsi utamanya adalah sebagai pintu pertukaran gas (mengambil CO2 untuk fotosintesis dan melepaskan O2) serta mengatur laju penguapan air (*transpirasi*) agar tanaman tidak layu saat suhu udara panas."
        )

    if any(k in msg_lower for k in ["transpirasi", "penguapan air"]):
        return (
            "**Transpirasi** adalah proses hilangnya uap air dari jaringan hidup tanaman melalui stomata daun ke atmosfer. "
            "Proses ini penting untuk menarik air dan unsur hara dari akar ke daun (*daya hisap daun*) serta mendinginkan suhu tubuh tanaman saat cuaca terik."
        )

    if any(k in msg_lower for k in ["respirasi", "pernapasan tumbuhan"]):
        return (
            "**Respirasi Tumbuhan** adalah proses pembakaran glukosa menggunakan oksigen untuk menghasilkan energi kimia (ATP), air, dan karbon dioksida. "
            "Respirasi berlangsung terus-menerus selama 24 jam di seluruh bagian sel hidup tanaman untuk mendukung pembelahan sel, penyerapan hara, dan pertumbuhan tunas baru."
        )

    if any(k in msg_lower for k in ["aeroponik", "aeroponik itu"]):
        return (
            "**Aeroponik** adalah sistem budidaya tanaman modern di mana akar tanaman dibiarkan menggantung di udara dalam ruang tertutup dan secara berkala disemprotkan kabut (*mist*) larutan nutrisi kaya oksigen. "
            "Sistem ini memberikan aerasi akar yang maksimal sehingga tanaman tumbuh lebih cepat dibanding metode konvensional."
        )

    if any(k in msg_lower for k in ["ph air", "ph tanah", "derajat keasaman"]):
        return (
            "**pH (Derajat Keasaman):**\n"
            "- Untuk budidaya pakcoy, pH ideal berada pada kisaran netral sedikit asam, yaitu **6.0 hingga 6.8**.\n"
            "- Jika pH terlalu asam (< 5.5) atau terlalu basa (> 7.5), unsur hara penting seperti Fosfor (P) dan Besi (Fe) akan terikat sehingga akar tidak dapat menyerapnya."
        )

    if any(k in msg_lower for k in ["ppm", "ec", "kepekatan nutrisi"]):
        return (
            "**PPM (Parts Per Million) / EC (Electrical Conductivity):**\n"
            "- Mengukur kepekatan larutan garam mineral/pupuk dalam air.\n"
            "- Untuk pakcoy hidroponik: fase semai membutuhkan **400–600 PPM**, fase vegetatif **800–1000 PPM**, dan fase pra-panen hingga **1050–1400 PPM** (EC 1.5–2.0)."
        )

    if "hidroponik" in msg_lower:
        return (
            "**Hidroponik** adalah metode budidaya menanam tanaman tanpa menggunakan media tanah, melainkan memanfaatkan air yang telah dicampur larutan nutrisi hara mineral esensial (seperti AB Mix). "
            "Contoh sistem hidroponik yang populer untuk pakcoy meliputi NFT (*Nutrient Film Technique*), DFT (*Deep Flow Technique*), dan sistem Wick (sumbu)."
        )

    if any(k in msg_lower for k in ["apa itu iot", "internet of things"]):
        return (
            "**Internet of Things (IoT)** adalah konsep teknologi di mana berbagai perangkat fisik (seperti sensor kelembaban tanah, sensor suhu, modul ESP32/Arduino, dan relay pompa) terhubung ke jaringan internet. "
            "Pada sistem **Smart Farming Pakcoy**, IoT memungkinkan penyiraman otomatis berjalan sendiri tanpa harus ditunggui petani secara manual."
        )

    if "lstm" in msg_lower:
        return (
            "**LSTM** (*Long Short-Term Memory*) adalah arsitektur kecerdasan buatan (Deep Learning) tipe Recurrent Neural Network (RNN) yang sangat andal dalam memproses data deret waktu (*time-series*). "
            "Dalam sistem ini, LSTM digunakan untuk memproyeksikan tren kelembapan tanah, suhu, dan pertumbuhan pakcoy 4 hari (12 timestep) ke depan."
        )

    if "cnn" in msg_lower:
        return (
            "**CNN** (*Convolutional Neural Network*) adalah arsitektur kecerdasan buatan Deep Learning untuk visi komputer (*computer vision*). "
            "Dalam aplikasi ini, CNN bertugas memverifikasi spesies daun (Pakcoy vs Selada vs Bayam), mendeteksi tahap kematangan, dan mengukur estimasi lebar daun secara otomatis dari foto kamera."
        )

    if any(k in msg_lower for k in ["hst", "dap", "apa itu hst", "arti hst"]):
        return (
            "**HST** adalah singkatan dari **Hari Setelah Tanam** (dalam istilah internasional dikenal sebagai **DAP** atau *Days After Planting*). "
            "HST adalah satuan hitungan umur tanaman yang dihitung sejak bibit pakcoy dipindahkan dari wadah persemaian ke media tanam utama hingga waktu panen tiba (0 hingga 40 HST)."
        )

    if "presiden indonesia" in msg_lower:
        return "Presiden Republik Indonesia saat ini adalah **Prabowo Subianto** (didampingi oleh Wakil Presiden Gibran Rakabuming Raka)."

    if "ibu kota indonesia" in msg_lower or "ibukota indonesia" in msg_lower:
        return "Ibu Kota Negara Indonesia saat ini adalah **Ibu Kota Nusantara (IKN)** yang terletak di Kalimantan Timur, dengan Jakarta sebagai Daerah Khusus Jakarta (DKJ)."

    if "suhu normal manusia" in msg_lower:
        return "Suhu tubuh normal manusia sehat berkisar antara **36,5°C hingga 37,2°C**."

    # =========================================================================
    # 6. STATUS SENSOR & KEBUN BATCH AKTIF SAAT INI
    # =========================================================================
    if any(k in msg_lower for k in ["kondisi", "sensor", "suhu", "kelembaban", "kelembapan", "pompa", "kebun", "status"]):
        pump_desc = "sedang **AKTIF (Menyiram)** untuk menaikkan kelembapan tanah" if pump == "ON" else "dalam posisi **STANDBY (OFF)** karena tanah masih cukup lembap"
        return (
            f"Berdasarkan sensor *real-time* untuk batch fokus aktif (**{batch_id}**):\n\n"
            f"- **Umur Tanaman:** **{dap} HST**\n"
            f"- **Kelembapan Tanah:** **{moisture:.1f}%** ({condition}, rentang ideal: 50%–65%)\n"
            f"- **Suhu Greenhouse:** **{temp:.1f}°C** (rentang ideal: 22°C–28°C)\n"
            f"- **Status Pompa Irigasi:** Otomatis {pump_desc}.\n"
            f"- **Lebar Daun:** **{leaf_w:.1f} cm** | **Kematangan:** **{mat_val:.1f}%**.\n\n"
            f"Ada pertanyaan lain seputar batch ini atau 40 batch lainnya di kebun?"
        )

    # =========================================================================
    # 7. SAPAAN & INTERAKSI RAMAH
    # =========================================================================
    if any(greet in msg_lower for greet in ["halo", "hai", "selamat", "assalamualaikum", "hi", "hey"]):
        return (
            f"Halo, selamat {time_greeting}! Saya adalah asisten virtual pintar **Smart Farming Pakcoy**. "
            f"Saya memantau seluruh 40 batch tanaman kebun (**PCK-001** s.d. **PCK-040**). "
            f"Saat ini batch fokus aktif adalah **{batch_id}** ({dap} HST, kelembapan {moisture:.1f}%, suhu {temp:.1f}°C). "
            f"Ada yang bisa saya bantu seputar budidaya, kondisi batch, atau pertanyaan lainnya hari ini?"
        )

    if any(k in msg_lower for k in ["siapa kamu", "apa tugasmu", "kamu siapa"]):
        return (
            "Saya adalah asisten virtual pintar untuk sistem **Smart Farming Pakcoy**. "
            "Tugas saya adalah membantu Anda memantau seluruh **40 batch tanaman** (**PCK-001** s.d. **PCK-040**), "
            "memberikan analisis telemetri sensorik *real-time* (kelembaban tanah, suhu, pompa otomatis), hasil AI (CNN & LSTM), "
            "serta menjawab segala pertanyaan seputar budidaya pakcoy maupun pengetahuan umum dengan ramah dan jelas!"
        )

    if any(k in msg_lower for k in ["apa kabar", "gimana kabarnya"]):
        return (
            f"Kabar saya sangat baik dan sistem monitoring 40 batch kebun pakcoy saat ini beroperasi lancar! "
            f"Batch aktif **{batch_id}** berada dalam kondisi {condition} ({moisture:.1f}% kelembapan, {temp:.1f}°C). Bagaimana dengan Anda hari ini?"
        )

    if any(k in msg_lower for k in ["terima kasih", "makasih", "thanks", "tengkyu"]):
        return "Sama-sama! Senang bisa membantu Anda. Jika ada hal lain yang ingin ditanyakan seputar tanaman atau topik lainnya, saya selalu siap membantu!"

    if any(k in msg_lower for k in ["bagus", "keren", "mantap", "hebat"]):
        return "Terima kasih atas apresiasinya! Saya akan terus siap membantu memantau kebun pakcoy dan memberikan informasi terbaik untuk Anda."

    # =========================================================================
    # 8. JAWABAN UMUM SEDERHANA & INFORMATIF
    # =========================================================================
    return (
        f"Pertanyaan Anda sangat menarik: *\"{last_user_msg}\"*. "
        f"Sebagai asisten pintar **Smart Farming Pakcoy**, saya siap membantu menjelaskan berbagai topik budidaya pakcoy, "
        f"memeriksa kondisi 40 batch kebun (**PCK-001** s.d. **PCK-040**), maupun panduan pertanian cerdas lainnya. "
        f"Silakan ajukan pertanyaan seputar fase pertumbuhan, hama, pemupukan, penyiraman, atau status panen tanaman!"
    )

def run_gemini_chat_query(
    messages: list,
    api_key: str = None,
    model: str = "gemini-3-flash-preview",
    sensor_context: dict = None,
    current_plant_id: str = "PCK-001"
) -> dict:
    """
    Menjalankan query percakapan ke Google Gemini API menggunakan SDK resmi google-genai.
    Model default: gemini-3-flash-preview (dengan fallback ke gemini-3.1-flash-lite / gemini-3.8-flash / gemini-flash-latest).
    Mendukung penuh 40 batch (PCK-001 s.d. PCK-040) dan semua pertanyaan sederhana.
    """
    key = api_key or get_gemini_api_key()
    last_user_msg = messages[-1]["content"] if messages else ""
    
    # Jika API key belum terisi, gunakan engine offline fallback yang kaya pengetahuan
    if not key or not key.strip():
        reply = generate_offline_fallback(messages, sensor_context, current_plant_id)
        return {"role": "assistant", "content": reply, "tool_calls": []}
    
    clean_key = key.strip()
    system_prompt = build_system_prompt(sensor_context, current_plant_id, user_query=last_user_msg)
    
    # 1. Format percakapan ke types.Content
    contents = []
    for m in messages:
        role = "user" if m.get("role") == "user" else "model"
        text = str(m.get("content", "")).strip()
        if not text:
            continue
        if not contents and role == "model":
            continue
        if contents and contents[-1].role == role:
            contents[-1].parts.append(types.Part.from_text(text=text))
        else:
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=text)]
            ))
            
    if not contents:
        contents = [types.Content(role="user", parts=[types.Part.from_text(text="Halo")])]

    # 2. Siapkan model target dengan fallback berjenjang
    target_model = (model or "gemini-3-flash-preview").replace("models/", "")
    if target_model in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-lite"]:
        target_model = "gemini-3-flash-preview"

    candidate_models = [target_model]
    for alt in [
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-lite-preview",
        "gemini-3.8-flash",
        "gemini-flash-latest",
        "gemini-3.5-flash"
    ]:
        if alt not in candidate_models:
            candidate_models.append(alt)

    # 3. Panggil google-genai SDK
    client = None
    try:
        client = genai.Client(api_key=clean_key)
    except Exception as e:
        print(f"Error initializing genai.Client: {e}")

    if client:
        for cand in candidate_models:
            try:
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.4,
                )
                response = client.models.generate_content(
                    model=cand,
                    contents=contents,
                    config=config
                )
                if response and response.text:
                    return {
                        "role": "assistant",
                        "content": response.text.strip(),
                        "tool_calls": []
                    }
            except ClientError as ce:
                err_text = ce.message if hasattr(ce, "message") else str(ce)
                if "API_KEY_INVALID" in err_text or "API key not valid" in err_text:
                    return {
                        "role": "assistant",
                        "content": "⚠️ **Kunci API Google Gemini Tidak Valid**: Kunci API yang dimasukkan tidak diterima oleh Google AI Studio. Silakan periksa atau masukkan kembali API Key yang benar pada panel konfigurasi di atas.",
                        "tool_calls": []
                    }
                continue
            except APIError as ae:
                err_text = ae.message if hasattr(ae, "message") else str(ae)
                if "RESOURCE_EXHAUSTED" in err_text or "429" in err_text:
                    fallback_text = generate_offline_fallback(messages, sensor_context, current_plant_id)
                    return {
                        "role": "assistant",
                        "content": f"{fallback_text}\n\n*(Catatan: Kuota panggilan Gemini API saat ini sedang penuh, respon dialihkan otomatis ke engine cerdas lokal)*",
                        "tool_calls": []
                    }
                continue
            except Exception as e:
                print(f"Percobaan model {cand} gagal: {e}")
                continue

    # 4. Fallback ke REST API langsung jika SDK mengalami kendala jaringan khusus
    rest_contents = []
    for c in contents:
        parts_data = [{"text": p.text} for p in c.parts if hasattr(p, "text") and p.text]
        if parts_data:
            rest_contents.append({"role": c.role, "parts": parts_data})
            
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": rest_contents,
        "generationConfig": {"temperature": 0.4}
    }
    headers = {"Content-Type": "application/json"}
    
    for cand in candidate_models:
        for api_ver in ["v1beta", "v1"]:
            rest_url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{cand}:generateContent?key={clean_key}"
            try:
                resp = requests.post(rest_url, json=payload, headers=headers, timeout=15)
                if resp.status_code == 200:
                    res_json = resp.json()
                    candidates = res_json.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text_parts = [p.get("text", "") for p in parts if "text" in p]
                        return {
                            "role": "assistant",
                            "content": "".join(text_parts).strip(),
                            "tool_calls": []
                        }
            except Exception:
                continue

    # 5. Fallback cerdas offline jika semua koneksi internet/API gagal
    fallback_text = generate_offline_fallback(messages, sensor_context, current_plant_id)
    return {
        "role": "assistant",
        "content": fallback_text,
        "tool_calls": []
    }
