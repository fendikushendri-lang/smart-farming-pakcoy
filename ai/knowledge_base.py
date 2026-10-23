"""
Knowledge Base khusus Tanaman Pakcoy (Brassica rapa subsp. chinensis)
Menyediakan referensi agronomis, parameter ideal, fase fisiologis, dan panduan budidaya.
Aturan Mutlak Budidaya: Masa panen pakcoy maksimal adalah 40 hari (HST / DAP).
"""

PACOY_KNOWLEDGE = {
    "nama_ilmiah": "Brassica rapa subsp. chinensis",
    "nama_lokal": ["Pakcoy", "Bok Choy", "Sawi Sendok", "Pokcoy"],
    "aturan_masa_panen": {
        "batas_maksimal_hst": 40,
        "jendela_panen_optimal": "30 - 38 HST",
        "batas_kritis_panen": "40 HST (Batas mutlak panen, > 40 HST memicu serat alot dan bolting)",
        "deskripsi": "Pakcoy adalah komoditas berumur genjah dengan siklus hidup panen maksimal 40 hari setelah tanam (HST)."
    },
    "fase_pertumbuhan": {
        "Initial Stage (Semai / Bibit)": {
            "rentang_hst": "1 - 10 HST (DAP)",
            "karakteristik": "Kotiledon muncul, 2-4 helai daun sejati kecil, perakaran muda mulai berkembang.",
            "kebutuhan_air": "Kelembaban media cukup tinggi (65-75%), drainase lancar, jangan tergenang.",
            "kebutuhan_suhu": "25 - 30°C optimal untuk perkecambahan dan translokasi fotosintat awal.",
            "tindakan_petani": "Pastikan pencahayaan cukup agar bibit tidak etiolasi (kutilang), jaga kelembaban tanah.",
            "sisa_ke_panen_maksimal": "30 - 39 hari lagi menuju batas maksimal 40 HST."
        },
        "Vegetative Stage (Masa Pertumbuhan Cepat)": {
            "rentang_hst": "11 - 25 HST (DAP)",
            "karakteristik": "Pembentukan roset daun aktif, daun melebar hijau segar, tangkai daun (petiole) mulai menebal.",
            "kebutuhan_air": "Kelembaban media optimal (55-65%), kebutuhan air meningkat seiring pertambahan biomassa.",
            "kebutuhan_suhu": "22 - 28°C untuk fotosintesis optimal, toleransi hingga 30°C.",
            "tindakan_petani": "Pemupukan unsur nitrogen (N) dan kalium (K), pantau kelembaban dan sirkulasi udara.",
            "sisa_ke_panen_maksimal": "15 - 29 hari lagi menuju batas maksimal 40 HST."
        },
        "Pre-Harvest Stage (Mendekati Panen)": {
            "rentang_hst": "26 - 34 HST (DAP)",
            "karakteristik": "Kanopi daun memadat, daun lebih kaku dan renyah, pertumbuhan vertikal melambat, akumulasi berat segar.",
            "kebutuhan_air": "Kelembaban terkontrol (50-55%), hindari over-watering agar tidak memicu busuk basah.",
            "kebutuhan_suhu": "24 - 28°C.",
            "tindakan_petani": "Kurangi intensitas penyiraman menjelang panen, periksa tanda-tanda kematangan panen.",
            "sisa_ke_panen_maksimal": "6 - 14 hari lagi menuju batas maksimal 40 HST."
        },
        "Harvest Ready (Siap Panen - Maksimal 40 Hari)": {
            "rentang_hst": "35 - 40 HST (DAP) [MAKSIMAL 40 HARI]",
            "karakteristik": "Roset tanaman sempurna, daun lebar mangkok seragam, pelepah tebal renyah, bobot optimal 150-250 gram. Harus dipanen maksimal pada 40 HST.",
            "kebutuhan_air": "Kelembaban stabil ~50%.",
            "kebutuhan_suhu": "Suhu ruang sejuk 22-26°C.",
            "tindakan_petani": "Segera panen pagi (sebelum pukul 09.00) atau sore. JANGAN tunda melewati 40 HST untuk mencegah penurunan mutu dan bolting.",
            "sisa_ke_panen_maksimal": "0 - 5 hari lagi (Batas akhir panen 40 HST)!"
        }
    },
    "parameter_lingkungan_ideal": {
        "suhu_optimal": "22°C - 28°C (Toleransi 20°C - 32°C)",
        "kelembaban_tanah_optimal": "50% - 65% (Kategori: Optimal)",
        "anomali_kelembaban_tinggi": "> 75% berisiko membusukkan akar / busuk lunak (Wet condition)",
        "anomali_kelembaban_rendah": "< 50% menyebabkan tanaman layu dan stomata menutup (Dry condition)",
        "ph_tanah_optimal": "6.0 - 6.8",
        "kebutuhan_cahaya": "6 - 8 jam sinar matahari penuh per hari"
    },
    "rekomendasi_tindakan": {
        "wet_soil": "Kurangi debit irigasi/penyiraman, periksa saluran aerasi dan drainase polybag/tray agar media tidak anaerob.",
        "dry_soil": "Lakukan penyiraman teratur pada pagi hari (08:00 WIB) atau sore (16:00 WIB) untuk mengembalikan kelembaban ke kisaran 50-60%.",
        "stagnant_growth": "Periksa kecukupan nutrisi (EC/PPM jika hidroponik atau pupuk NPK), pastikan aerasi akar tidak terhambat.",
        "harvest_urgent": "🚨 PERINGATAN PANEN: Umur tanaman telah mencapai/melebihi batas maksimal panen 40 HST. Segera panen hari ini!",
        "harvest_ready": "🥬 JENDELA PANEN OPTIMAL: Tanaman siap dipanen (umur 35-40 HST). Jadwalkan pemanenan sebelum melewati batas 40 HST.",
        "optimal": "Pertahankan parameter saat ini. Tanaman berada dalam kondisi fisiologis prima dalam siklus panen maksimal 40 HST."
    }
}

def get_growth_stage_info(stage_name: str) -> dict:
    """Mengembalikan data agronomi untuk fase pertumbuhan tertentu."""
    for k, v in PACOY_KNOWLEDGE["fase_pertumbuhan"].items():
        if stage_name.lower() in k.lower():
            return v
    return PACOY_KNOWLEDGE["fase_pertumbuhan"]["Vegetative Stage (Masa Pertumbuhan Cepat)"]

def get_agronomic_recommendation(soil_moisture: float, temperature: float, dap: int = None) -> list:
    """Mengembalikan daftar rekomendasi agronomi berbasis kelembaban, suhu, dan batas panen 40 HST."""
    recs = []
    
    # Check Harvest Deadline (Maksimal 40 Hari)
    if dap is not None:
        if dap >= 40:
            recs.append("🚨 <strong>BATAS PANEN MAKSIMAL (40 HST):</strong> Tanaman telah mencapai umur maksimal 40 hari. Segera panen hari ini untuk mencegah serat mengeras dan risiko berbunga (bolting).")
        elif dap >= 35:
            recs.append(f"🥬 <strong>JENDELA PANEN OPTIMAL ({dap}/40 HST):</strong> Tanaman telah siap panen. Sisa {40 - dap} hari menuju batas maksimal 40 HST.")
            
    if soil_moisture > 70.0:
        recs.append(PACOY_KNOWLEDGE["rekomendasi_tindakan"]["wet_soil"])
    elif soil_moisture < 50.0:
        recs.append(PACOY_KNOWLEDGE["rekomendasi_tindakan"]["dry_soil"])
    else:
        recs.append(PACOY_KNOWLEDGE["rekomendasi_tindakan"]["optimal"])
    
    if temperature > 29.0:
        recs.append("Tingkatkan sirkulasi udara atau pasang ventilasi teduh untuk mencegah transpirasi daun berlebih.")
    elif temperature < 20.0:
        recs.append("Pertahankan suhu mikro di atas 20°C untuk menjaga laju metabolisme fotosintesis optimal.")
        
    return recs
