import os
import base64

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, "data", "images")

PLANT_IMAGES = {
    "PCK-001": os.path.join(IMAGES_DIR, "Siap_Panen", "Siap_Panen_073.jpg"),
    "PCK-002": os.path.join(IMAGES_DIR, "Masa_Pertumbuhan", "Masa_Pertumbuhan_008.jpg"),
    "PCK-003": os.path.join(IMAGES_DIR, "Mendekati_Panen", "Mendekati_Panen_038.jpg"),
    "PCK-004": os.path.join(IMAGES_DIR, "Siap_Panen", "Siap_Panen_074.jpg"),
    "PCK-005": os.path.join(IMAGES_DIR, "Masa_Pertumbuhan", "Masa_Pertumbuhan_001.jpg"),
    "PCK-FULL40": os.path.join(IMAGES_DIR, "Siap_Panen", "Siap_Panen_073.jpg")
}

def get_image_base64(plant_id: str):
    path = PLANT_IMAGES.get(plant_id)
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"
            
    # Fallback to any first image if not found
    fallback = os.path.join(IMAGES_DIR, "Siap_Panen", "Siap_Panen_073.jpg")
    if os.path.exists(fallback):
        with open(fallback, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"
    return ""
