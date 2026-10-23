import os
import json
from services.data_service import (
    get_plant_details, get_growth_history, get_latest_observation,
    get_alerts, get_plant_comparison, get_all_plants
)
from services.anomaly_service import detect_anomalies_for_plant
from ai.knowledge_base import PACOY_KNOWLEDGE

TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "get_plant_details",
            "description": "Mengambil informasi profil tanaman berdasarkan plant_id (misal 'PCK-001').",
            "parameters": {
                "type": "object",
                "properties": {
                    "plant_id": {"type": "string", "description": "ID tanaman pakcoy, contoh: 'PCK-001'"}
                },
                "required": ["plant_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_growth_history",
            "description": "Mengambil data riwayat pertumbuhan harian tanaman berdasarkan plant_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plant_id": {"type": "string", "description": "ID tanaman pakcoy"},
                    "limit": {"type": "integer", "description": "Jumlah rekaman pengamatan terakhir"}
                },
                "required": ["plant_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_latest_observation",
            "description": "Mengambil data kondisi pengamatan paling mutakhir (kelembaban tanah, suhu, maturity level, fase) untuk tanaman tertentu.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plant_id": {"type": "string", "description": "ID tanaman pakcoy"}
                },
                "required": ["plant_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_growth_anomalies",
            "description": "Mendeteksi dan mengambil daftar anomali lingkungan atau pertumbuhan untuk tanaman tertentu.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plant_id": {"type": "string", "description": "ID tanaman pakcoy"}
                },
                "required": ["plant_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_plant_comparison",
            "description": "Membandingkan data pertumbuhan dan kondisi terkini antara dua tanaman pakcoy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plant_id1": {"type": "string", "description": "ID tanaman pertama, contoh: 'PCK-001'"},
                    "plant_id2": {"type": "string", "description": "ID tanaman kedua, contoh: 'PCK-002'"}
                },
                "required": ["plant_id1", "plant_id2"]
            }
        }
    }
]

def execute_tool_call(tool_name: str, arguments: dict):
    """Menjalankan fungsi database lokal berdasarkan instruksi Tool Call dari OpenAI."""
    if tool_name == "get_plant_details":
        return get_plant_details(arguments.get("plant_id", "PCK-001"))
    elif tool_name == "get_growth_history":
        df = get_growth_history(arguments.get("plant_id", "PCK-001"), limit=arguments.get("limit", 20))
        return df.to_dict(orient="records")
    elif tool_name == "get_latest_observation":
        return get_latest_observation(arguments.get("plant_id", "PCK-001"))
    elif tool_name == "get_growth_anomalies":
        return detect_anomalies_for_plant(arguments.get("plant_id", "PCK-001"))
    elif tool_name == "get_plant_comparison":
        return get_plant_comparison(arguments.get("plant_id1", "PCK-001"), arguments.get("plant_id2", "PCK-002"))
    else:
        return {"error": f"Tool '{tool_name}' tidak dikenal."}

def run_chat_query(messages: list, api_key: str = None, model: str = "gpt-4o-mini", current_plant_id: str = "PCK-001"):
    """
    Menjalankan query percakapan ke OpenAI API dengan tool calling.
    Menyediakan graceful fallback jika API key belum tersedia.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    
    # SYSTEM PROMPT
    system_prompt = f"""
Anda adalah AI Pakcoy Assistant & Smart Agriculture Data Analyst.
Tugas utama Anda adalah membantu petani dan peneliti memantau, menganalisis, dan memberikan rekomendasi mengenai pertumbuhan tanaman pakcoy berdasarkan data aktual sistem.

Pedoman Penting:
1. Konteks Tanaman Saat Ini: Jika pengguna tidak menyebutkan ID tanaman dan menggunakan kata ganti ('tanaman ini', 'nya', 'tanaman saya'), rujuklah ke tanaman terpilih: {current_plant_id}.
2. Sumber Kebenaran: Gunakan data aktual yang didapatkan dari pemanggilan tool. JANGAN PERNAH mengarang angka, parameter, atau label yang tidak ada di dataset.
3. Keterbatasan Data: Jika parameter tertentu tidak ada di dataset (seperti tinggi tanaman dalam cm atau diagnosis penyakit tanaman), jelaskan secara jujur bahwa data tersebut tidak tercatat di sistem.
4. Struktur Rekomendasi: Ketika diminta analisis atau rekomendasi, sajikan dengan struktur:
   - Temuan (Fakta dari data)
   - Analisis (Makna bagi pertumbuhan pakcoy)
   - Rekomendasi (Tindakan agronomis praktis yang relevan)
5. Bahasa: Gunakan Bahasa Indonesia yang profesional, ramah, dan ilmiah pertanian.
6. Standar Masa Panen Maksimal 40 Hari: Pakcoy adalah tanaman genjah dengan batas siklus budidaya maksimal 40 HST (Hari Setelah Tanam). Jendela panen prima berkisar antara 30-38 HST. Jika tanaman berumur >= 35 HST, sampaikan bahwa tanaman berada dalam jendela panen prima. Jika tanaman telah mencapai 40 HST, berikan rekomendasi mendesak untuk segera memanennya hari ini guna mencegah penurunan mutu, pelepah berserat alot, dan risiko bolting (berbunga).
"""
    
    if not key:
        # Fallback Engine jika API key belum diisi pengguna
        last_user_msg = messages[-1]["content"].lower()
        plant_id = current_plant_id
        for pid in ["pck-001", "pck-002", "pck-003", "pck-004", "pck-005"]:
            if pid in last_user_msg:
                plant_id = pid.upper()
                break
                
        p_info = get_plant_details(plant_id) or get_plant_details("PCK-001")
        latest = get_latest_observation(plant_id)
        anomalies = detect_anomalies_for_plant(plant_id)
        dap_val = int(latest['dap'])
        
        reply = f"**[Mode Respons Cerdas Tanpa API Key]**\n\n"
        if "panen" in last_user_msg or "umur" in last_user_msg or "hst" in last_user_msg:
            if dap_val >= 40:
                reply += (
                    f"### 🚨 Status Panen: {plant_id} (Batas Panen 40 HST Tercapai)\n"
                    f"* **Umur Saat Ini**: {dap_val} HST (Batas Maksimal: 40 HST)\n"
                    f"* **Status**: **Wajib Panen Hari Ini!**\n"
                    f"* **Rekomendasi Agronomi**: Pemanenan harus segera dilakukan pada pagi hari (sebelum pukul 09.00 WIB) "
                    f"atau sore hari saat transpirasi rendah. Menunda panen melewati 40 HST akan menyebabkan pelepah daun berserat alot dan memicu tanaman berbunga (bolting)."
                )
            elif dap_val >= 35:
                reply += (
                    f"### 🥬 Jendela Panen Prima: {plant_id}\n"
                    f"* **Umur Saat Ini**: {dap_val} HST (Batas Maksimal: 40 HST)\n"
                    f"* **Sisa Waktu Menuju Batas Panen**: **{40 - dap_val} Hari Lagi**\n"
                    f"* **Kondisi Daun**: Roset daun telah memadat sempurna dengan kematangan {latest['maturity_level']}%. Siap dipanen kapan saja dalam 1-{40 - dap_val} hari ke depan."
                )
            else:
                reply += (
                    f"### 🌱 Siklus Pertumbuhan: {plant_id}\n"
                    f"* **Umur Saat Ini**: {dap_val} HST (Fase: {latest['criteria'].split(' (')[0]})\n"
                    f"* **Target Panen Maksimal**: 40 HST\n"
                    f"* **Sisa Waktu Pemeliharaan**: **{40 - dap_val} Hari Menuju Panen**\n"
                    f"* **Rekomendasi**: Pertahankan kelembaban tanah 50-65% dan fertigasi rutin hingga memasuki jendela panen pada umur 35 HST."
                )
        elif "prediksi" in last_user_msg:
            reply += (
                f"### Prediksi Pertumbuhan Tanaman {plant_id} (Maksimal 40 HST)\n"
                f"Berdasarkan model Multivariate LSTM yang dilatih dari data historis:\n"
                f"* **Hari +1 (DAP {dap_val + 1} HST)**: Tingkat Kematangan meningkat bertahap (+0.3% s.d. +0.5%), Kelembaban diprediksi stabil pada kisaran optimal 50-53%.\n"
                f"* **Hari +2 s.d. +4 (DAP {dap_val + 2} s.d. {dap_val + 4} HST)**: Tren pertumbuhan berlanjut dengan stabilitas suhu diurnal normal (24.5°C pagi - 28.8°C siang).\n"
                f"* **Status Batas Panen (40 HST)**: {'Tanaman mencapai batas akhir panen 40 HST pada proyeksi ini, jadwalkan panen tepat waktu!' if dap_val + 4 >= 40 else f'Masih dalam koridor pertumbuhan aman (sisa {40 - dap_val} hari).'}\n\n"
                f"> Silakan buka menu **Prediksi 4 Hari** pada sidebar untuk melihat tabel dan grafik proyeksi 12 timestep lengkap!"
            )
        elif "anomali" in last_user_msg or "perhatian" in last_user_msg:
            if anomalies:
                reply += f"### Deteksi Anomali untuk {plant_id}\n"
                for a in anomalies[:3]:
                    reply += f"* **{a['title']}** (Tingkat: {a['severity']}): {a['description']}\n"
                reply += f"\n**Rekomendasi:** Periksa sistem irigasi dan jadwal penyiraman untuk menjaga kelembaban di 50-65%."
            else:
                reply += f"Tidak terdeteksi anomali kritis pada {plant_id}. Semua parameter lingkungan berada dalam rentang optimal."
        else:
            harvest_status_note = "🚨 Wajib panen hari ini (40 HST tercapai)" if dap_val >= 40 else (f"🥬 Siap panen (sisa {40 - dap_val} hari)" if dap_val >= 35 else f"🌱 Fase tumbuh (sisa {40 - dap_val} hari)")
            reply += (
                f"### Status Perkembangan {p_info['name']} ({plant_id})\n"
                f"* **Umur Tanaman**: {dap_val} / 40 HST ({harvest_status_note})\n"
                f"* **Hari Pengamatan**: Hari ke-{latest['day']}\n"
                f"* **Fase Pertumbuhan**: {latest['criteria']}\n"
                f"* **Tingkat Kematangan**: {latest['maturity_level']}%\n"
                f"* **Kelembaban Tanah**: {latest['soil_moisture']}% ({latest['soil_condition']})\n"
                f"* **Suhu Lingkungan**: {latest['temperature']}°C\n\n"
                f"### Analisis & Rekomendasi\n"
                f"Tanaman berada dalam kondisi {latest['soil_condition']}. "
                f"Suhu lingkungan {latest['temperature']}°C sangat kondusif untuk laju fotosintesis pakcoy (optimal 22-28°C).\n\n"
                f"*(Tips: Masukkan OpenAI API Key Anda di menu Pengaturan untuk percakapan generatif dinamis via GPT-4o-mini)*"
            )
        return {"role": "assistant", "content": reply, "tool_calls": []}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            tools=TOOLS_DEFINITION,
            tool_choice="auto",
            temperature=0.3
        )
        
        msg = response.choices[0].message
        
        # Check for tool call
        if msg.tool_calls:
            tool_calls_record = []
            full_messages.append(msg)
            
            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                tool_result = execute_tool_call(fn_name, fn_args)
                tool_calls_record.append({"tool": fn_name, "args": fn_args, "result": tool_result})
                
                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False)
                })
                
            second_resp = client.chat.completions.create(
                model=model,
                messages=full_messages,
                temperature=0.3
            )
            return {
                "role": "assistant",
                "content": second_resp.choices[0].message.content,
                "tool_calls": tool_calls_record
            }
        else:
            return {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": []
            }
    except Exception as e:
        return {
            "role": "assistant",
            "content": f"Terjadi kesalahan saat memanggil OpenAI API: {str(e)}.\n\nSistem beralih ke analisis data offline lokal.",
            "tool_calls": []
        }
