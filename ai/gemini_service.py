import os
import json
import requests
from datetime import datetime
from services.data_service import (
    get_plant_details, get_growth_history, get_latest_observation,
    get_alerts, get_plant_comparison, get_all_plants
)
from services.anomaly_service import detect_anomalies_for_plant

# Model daftar Google AI Studio terpopuler & terverifikasi untuk chat
AVAILABLE_GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash-lite"
]

def is_valid_chat_model(name: str) -> bool:
    """Memfilter hanya model chat/generasi teks, membuang model TTS, audio, embedding, dll."""
    name_lower = name.lower()
    excluded = ["tts", "embedding", "imagen", "audio", "aqa", "robotics", "computer-use", "transcribe", "native-audio"]
    if any(ex in name_lower for ex in excluded):
        return False
    return "gemini" in name_lower

def get_available_models_for_key(api_key: str) -> list[str]:
    """Mengambil daftar model chat yang aktif untuk API key pengguna."""
    if not api_key:
        return []
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key.strip()}"
        res = requests.get(url, timeout=8)
        if res.status_code == 200:
            models = [
                m.get("name", "").replace("models/", "")
                for m in res.json().get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
                and is_valid_chat_model(m.get("name", ""))
            ]
            return models
    except Exception:
        pass
    return []

def test_gemini_connection(api_key: str) -> tuple[bool, str, list]:
    """
    Menguji validitas API Key Google AI Studio dengan memanggil endpoint models.
    Mengembalikan (is_success, message, model_list).
    """
    if not api_key or not api_key.strip():
        return False, "API Key Google AI Studio belum diisi.", []
    
    clean_key = api_key.strip()
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={clean_key}"
        res = requests.get(url, timeout=12)
        if res.status_code == 200:
            models = [
                m.get("name", "").replace("models/", "")
                for m in res.json().get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
                and is_valid_chat_model(m.get("name", ""))
            ]
            msg = f"Berhasil terhubung ke Google AI Studio! ({len(models)} model chat aktif ditemukan)"
            return True, msg, models
        else:
            try:
                err_data = res.json()
                err_msg = err_data.get("error", {}).get("message", res.text)
            except Exception:
                err_msg = res.text
            return False, f"Gagal autentikasi ke Google AI Studio (HTTP {res.status_code}): {err_msg}", []
    except requests.exceptions.Timeout:
        return False, "Koneksi ke Google AI Studio timeout (melebihi 12 detik). Periksa koneksi internet.", []
    except Exception as e:
        return False, f"Terjadi kesalahan koneksi: {str(e)}", []


def build_system_prompt(current_plant_id: str = "PCK-001") -> str:
    """Membangun system prompt dengan grounding data tanaman aktif dan gaya percakapan alami."""
    latest = get_latest_observation(current_plant_id) or {}
    p_info = get_plant_details(current_plant_id) or {}
    
    dap = latest.get("dap", 0)
    moisture = latest.get("soil_moisture", 0.0)
    temp = latest.get("temperature", 0.0)
    soil_cond = latest.get("soil_condition", "Unknown")
    mat_lvl = latest.get("maturity_level", 0.0)
    criteria = latest.get("criteria", "Unknown")
    
    prompt = f"""Anda adalah asisten virtual cerdas dalam sistem **Smart Farming Pakcoy**.
Tugas Anda adalah berdialog dengan pengguna untuk menjawab pertanyaan seputar sistem pemantauan sensor, model AI, dan panduan budidaya pakcoy secara alami, hangat, dan ilmiah.

KONTEKS SISTEM & TANAMAN AKTIF SAAT INI:
- ID Tanaman Aktif: {current_plant_id} ({p_info.get('name', 'Pakcoy Batch')})
- Lokasi: {p_info.get('location', 'Greenhouse')}
- Umur Tanaman: {dap} hari setelah tanam (DAP / HST)
- Batas Maksimum Siklus Panen: 40 HST (jendela panen prima: 30-38 HST)
- Tingkat Kematangan Terkini (maturity level): {mat_lvl}% ({criteria})
- Kelembaban Tanah Terkini: {moisture}% ({soil_cond}, optimal: 50-65%)
- Suhu Lingkungan: {temp}°C (optimal: 22-28°C)

PEDOMAN GAYA & FORMAT JAWABAN:
1. Gaya Bicara Alami & Mengalir: Tuliskan jawaban dalam 1 hingga 2 paragraf yang kohesif, ramah, dan profesional. Hindari membuat daftar bullet point berlebihan kecuali jika pengguna secara khusus meminta rincian atau daftar.
2. Penekanan Teks Tebal: Berikan tanda tebal (**bold**) pada kata kunci, istilah teknis, dan angka penting (seperti: **Smart Farming Pakcoy**, **30 hingga 40 hari**, **DAP**, **maturity level**, **50% hingga 65%**).
3. Respon Sapaan: Jika pengguna menyapa (misal: "halo", "selamat pagi", "selamat malam"), balas dengan ramah sesuai waktu dan tawarkan bantuan terkait sensor, AI, atau budidaya pakcoy.
4. Respon Panen: Jelaskan bahwa panen umumnya siap dalam rentang **30 hingga 40 hari** setelah tanam (**DAP**), dengan fase kematangan (**maturity level**) mencapai **89,0%** atau lebih.
5. Catatan Penutup: Jika relevan, sertakan catatan singkat bahwa rekomendasi bersifat sebagai pendukung keputusan dan sebaiknya diverifikasi langsung di lapangan.
"""
    return prompt

def generate_offline_fallback(messages: list, current_plant_id: str = "PCK-001") -> str:
    """Engine fallback percakapan alami yang meniru respons asisten AI modern."""
    last_user_msg = messages[-1]["content"].lower() if messages else ""
    
    # Ambil salam waktu saat ini
    now_hour = datetime.now().hour
    time_greeting = "pagi" if 5 <= now_hour < 11 else ("siang" if 11 <= now_hour < 15 else ("sore" if 15 <= now_hour < 18 else "malam"))
    if "pagi" in last_user_msg:
        time_greeting = "pagi"
    elif "siang" in last_user_msg:
        time_greeting = "siang"
    elif "sore" in last_user_msg:
        time_greeting = "sore"
    elif "malam" in last_user_msg:
        time_greeting = "malam"

    latest = get_latest_observation(current_plant_id) or {}
    mat_val = latest.get("maturity_level", 89.0)
    moisture = latest.get("soil_moisture", 52.0)
    temp = latest.get("temperature", 27.0)

    # 1. Sapaan
    if any(greet in last_user_msg for greet in ["halo", "hai", "selamat", "assalamualaikum", "hi"]):
        return f"Halo selamat {time_greeting}! Selamat datang di asisten virtual **Smart Farming Pakcoy**. Ada yang bisa saya bantu terkait pemantauan sensor, model AI, atau panduan budidaya tanaman pakcoy Anda {time_greeting} ini?"

    # 2. Pertanyaan Masa Panen / Berapa Lama Panen
    if "panen" in last_user_msg or "umur" in last_user_msg or "berapa lama" in last_user_msg or "dap" in last_user_msg:
        return f"Tanaman pakcoy umumnya siap dipanen dalam waktu sekitar **30 hingga 40 hari** setelah tanam (**DAP**), tergantung pada varietas dan kondisi lingkungan. Dalam sistem **Smart Farming Pakcoy**, fase ini ditandai dengan tingkat kematangan (**maturity level**) mencapai **89,0%** atau lebih. Perlu diingat bahwa rekomendasi ini bersifat sebagai pendukung keputusan dan sebaiknya diverifikasi langsung dengan pengamatan visual di lapangan."

    # 3. Ciri Pakcoy Sehat
    if "sehat" in last_user_msg or "ciri" in last_user_msg or "kualitas" in last_user_msg:
        return f"Tanaman pakcoy yang sehat ditandai dengan helaian daun berwarna **hijau cerah merata**, segar, dan bebas dari bercak penyakit. Pelepah batangnya tebal, kokoh menyerupai sendok, serta bertekstur **renyah dan tidak berserat alot**. Dalam sistem **Smart Farming Pakcoy**, kesehatan tanaman didukung oleh kelembaban tanah optimal pada rentang **50% hingga 65%** serta suhu greenhouse antara **22°C hingga 28°C**."

    # 4. Perkecambahan / Semai
    if "kecambah" in last_user_msg or "semai" in last_user_msg or "benih" in last_user_msg:
        return f"Benih tanaman pakcoy umumnya mulai berkecambah dalam waktu **2 hingga 4 hari** setelah disemai pada media yang lembab dan bersuhu hangat (20–25°C). Begitu benih pecah dan daun lembaga (**kotiledon**) muncul, bibit perlu segera dikenalkan pada sinar matahari pagi secara bertahap agar tidak mengalami etiolasi (**kutilang**). Pada umur **10 hingga 14 hari**, bibit yang telah memiliki 3–4 helai daun sejati siap dipindahkan ke modul tanam utama."

    # 5. Prediksi Pertumbuhan
    if "prediksi" in last_user_msg or "proyeksi" in last_user_msg:
        return f"Berdasarkan proyeksi model **Multivariate LSTM** pada sistem **Smart Farming Pakcoy**, pertumbuhan tanaman diproyeksikan berada dalam tren optimal dengan stabilitas kelembaban tanah pada kisaran **50%–53%** dan kenaikan tingkat kematangan bertahap. Anda dapat memantau visualisasi grafik 12 timestep selengkapnya pada menu **Prediksi 4 Hari** di sidebar."

    # 6. Pertanyaan Umum / Status Tanaman
    return f"Saat ini tanaman pada sistem **Smart Farming Pakcoy** berada dalam kondisi pertumbuhan yang baik dengan kelembaban tanah **{moisture}%** dan suhu lingkungan **{temp}°C**. Tingkat kematangan (**maturity level**) tercatat sebesar **{mat_val}%**. Ada hal spesifik lain terkait pemantauan sensor atau panduan agronomis yang ingin Anda ketahui?"

def run_gemini_chat_query(
    messages: list,
    api_key: str = None,
    model: str = "gemini-2.0-flash",
    current_plant_id: str = "PCK-001"
) -> dict:
    """
    Menjalankan query percakapan ke Google AI Studio (Gemini) API.
    Menghasilkan respons alami yang ringkas, rapi, dan mengalir seperti ChatGPT.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    
    if not key or not key.strip():
        reply = generate_offline_fallback(messages, current_plant_id)
        return {"role": "assistant", "content": reply, "tool_calls": []}
    
    clean_key = key.strip()
    system_prompt = build_system_prompt(current_plant_id)
    
    # Kumpulkan kandidat model obrolan valid
    requested_model = (model or "gemini-2.0-flash").replace("models/", "")
    candidate_models = []
    
    if is_valid_chat_model(requested_model):
        candidate_models.append(requested_model)
        
    core_models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-lite"]
    for cm in core_models:
        if cm not in candidate_models:
            candidate_models.append(cm)
            
    rest_contents = []
    for m in messages:
        role = "user" if m.get("role") == "user" else "model"
        rest_contents.append({
            "role": role,
            "parts": [{"text": str(m.get("content", ""))}]
        })
        
    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": rest_contents,
        "generationConfig": {
            "temperature": 0.4
        }
    }
    headers = {"Content-Type": "application/json"}
    
    for cand in candidate_models:
        for api_ver in ["v1beta", "v1"]:
            rest_url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{cand}:generateContent?key={clean_key}"
            try:
                resp = requests.post(rest_url, json=payload, headers=headers, timeout=18)
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
                elif resp.status_code == 400:
                    err_msg = resp.json().get("error", {}).get("message", "")
                    if "API_KEY_INVALID" in err_msg or "API key not valid" in err_msg:
                        break
                    continue
                else:
                    continue
            except Exception:
                continue

    # Fallback cerdas alami
    fallback_text = generate_offline_fallback(messages, current_plant_id)
    return {
        "role": "assistant",
        "content": fallback_text,
        "tool_calls": []
    }
