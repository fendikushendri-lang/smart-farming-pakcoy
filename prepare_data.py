import os
import zipfile
import hashlib
import io
import shutil
import pandas as pd
from PIL import Image

BASE_DIR = r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new"
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

zip_mapping = {
    "Masa_Pertumbuhan": os.path.join(BASE_DIR, "s3Masa Pertumbuhan.zip"),
    "Mendekati_Panen": os.path.join(BASE_DIR, "s3Mendekati Panen pokcoy.zip"),
    "Siap_Panen": os.path.join(BASE_DIR, "s3siap panen pokcoy.zip")
}

print("=== 1. EXTRACTING & DEDUPLICATING IMAGES ===")
seen_hashes = {}
image_records = []

for label, z_path in zip_mapping.items():
    class_dir = os.path.join(IMAGES_DIR, label)
    os.makedirs(class_dir, exist_ok=True)
    
    with zipfile.ZipFile(z_path, 'r') as z:
        for fname in z.namelist():
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) and not fname.startswith('__MACOSX'):
                raw = z.read(fname)
                md5 = hashlib.md5(raw).hexdigest()
                
                # Deduplication check
                if md5 in seen_hashes:
                    print(f"Skipping duplicate: {fname} (matches {seen_hashes[md5]})")
                    continue
                
                # Check valid image
                try:
                    img = Image.open(io.BytesIO(raw))
                    img.verify()
                    # Reopen after verify
                    img = Image.open(io.BytesIO(raw))
                    clean_name = f"{label}_{len(image_records)+1:03d}.jpg"
                    save_path = os.path.join(class_dir, clean_name)
                    
                    # Convert to RGB and save as standard JPEG
                    rgb_img = img.convert('RGB')
                    rgb_img.save(save_path, "JPEG", quality=95)
                    
                    seen_hashes[md5] = clean_name
                    image_records.append({
                        "filename": clean_name,
                        "class": label,
                        "path": save_path,
                        "original_name": os.path.basename(fname),
                        "width": img.width,
                        "height": img.height
                    })
                except Exception as e:
                    print(f"Corrupt or invalid image skipped: {fname} ({e})")

img_meta_df = pd.DataFrame(image_records)
img_meta_df.to_csv(os.path.join(PROCESSED_DIR, "images_metadata.csv"), index=False)
print(f"Extracted {len(img_meta_df)} unique clean images.")
print(img_meta_df['class'].value_counts())

print("\n=== 2. PROCESSING EXCEL TIME SERIES ===")
excel_path = os.path.join(BASE_DIR, "new Dataset_Pakcoy.xlsx")
df = pd.read_excel(excel_path)

# Normalize column names
df.columns = [
    'day', 'dap', 'time', 'soil_moisture', 'temperature', 
    'soil_condition', 'maturity_level', 'criteria'
]

# Ensure correct types
df['day'] = df['day'].astype(int)
df['dap'] = df['dap'].astype(int)
df['soil_moisture'] = df['soil_moisture'].astype(float)
df['temperature'] = df['temperature'].astype(float)
df['maturity_level'] = df['maturity_level'].astype(float)

# Add sequential timestep index (0 to 1139)
df['timestep'] = range(len(df))

# Map criteria to clean growth stage
stage_map = {
    'Initial Stage (HST + Morphological Observation)': 'Initial Stage',
    'Vegetative Stage (HST + Morphological Observation)': 'Vegetative Stage',
    'Pre-Harvest Stage (HST + Morphological Observation)': 'Pre-Harvest Stage',
    'Harvest Ready (HST + Morphological Observation)': 'Harvest Ready'
}
df['growth_stage'] = df['criteria'].map(stage_map)

# Also create realistic plant batch partitions for monitoring
# e.g., Batch 1 (Days 1-35), Batch 2 (Days 36-70), etc., or Plant IDs PCK-001, PCK-002, etc.
# Pakcoy life cycle is typically 35-40 days.
# Let's map continuous days into realistic Pakcoy Batches / Plants:
# PCK-001: Days 1-35 (DAP 10-44, Full normal lifecycle from Initial to Harvest Ready)
# PCK-002: Days 36-70 (Simulated Cycle 2)
# PCK-003: Days 71-105 (Cycle 3)
# PCK-004: Days 106-140 (Cycle 4)
# ... up to PCK-010
# Also keep continuous dataset for LSTM time-series training.
df.to_csv(os.path.join(PROCESSED_DIR, "timeseries_clean.csv"), index=False)
print(f"Processed {len(df)} rows of time-series data saved to timeseries_clean.csv.")
print("Summary stats:")
print(df[['soil_moisture', 'temperature', 'maturity_level']].describe())
