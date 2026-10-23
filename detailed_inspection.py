import pandas as pd
import zipfile
import os
import io
from PIL import Image

excel_path = r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new\new Dataset_Pakcoy.xlsx"
df = pd.read_excel(excel_path)

print("="*60)
print("EXCEL DETAILED ANALYSIS")
print("="*60)
print(f"Total Rows: {len(df)}")
print(f"Total Columns: {len(df.columns)}")
print("Columns:", df.columns.tolist())

# Rename temperature column if it has corrupted character
temp_col = [c for c in df.columns if 'Temperature' in c][0]

print("\n--- Summary Statistics (Numeric) ---")
print(df.describe())

print("\n--- Unique values per column ---")
for col in df.columns:
    n_unique = df[col].nunique()
    print(f"\nColumn: '{col}' (dtype: {df[col].dtype}) - Unique count: {n_unique}")
    if n_unique <= 20:
        print("Value counts:\n", df[col].value_counts())
    else:
        print("Sample unique values:", df[col].unique()[:10])

print("\n--- Check DAP vs Day ---")
print("Min Day:", df['Day'].min(), "Max Day:", df['Day'].max())
print("Min DAP:", df['DAP'].min(), "Max DAP:", df['DAP'].max())
print("Relationship between Day and DAP (first 25 rows):")
print(df[['Day', 'DAP', 'Time', 'Soil Moisture (%)', temp_col, 'Soil Condition', 'Ground Truth Maturity Level (%)', 'Ground Truth Criteria']].head(25))

print("\n--- Unique Times per Day ---")
print(df.groupby('Day')['Time'].count().value_counts())
print("Sample Time values:", df['Time'].value_counts())

print("\n--- Grouped by Ground Truth Criteria ---")
print(df.groupby('Ground Truth Criteria')[['DAP', 'Soil Moisture (%)', temp_col, 'Ground Truth Maturity Level (%)']].agg(['min', 'mean', 'max', 'count']))

print("\n--- Grouped by Soil Condition ---")
print(df.groupby('Soil Condition')[['Soil Moisture (%)', temp_col, 'Ground Truth Maturity Level (%)']].agg(['min', 'mean', 'max', 'count']))

print("\n--- Check Day / DAP transition points ---")
# See if DAP resets or increments
print("Unique DAP values:", sorted(df['DAP'].unique()))
print("Does DAP match Day? Or does DAP cycle (e.g. planting cycles)?")
# Let's check how DAP changes across rows
print(df[['Day', 'DAP']].drop_duplicates().head(50))

print("\n" + "="*60)
print("ZIP FILES IMAGE DETAILED ANALYSIS")
print("="*60)

zip_files = {
    "Masa Pertumbuhan": r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new\s3Masa Pertumbuhan.zip",
    "Mendekati Panen": r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new\s3Mendekati Panen pokcoy.zip",
    "Siap Panen": r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new\s3siap panen pokcoy.zip"
}

total_valid_imgs = 0
for label, z_path in zip_files.items():
    print(f"\n--- Class / Label: {label} ---")
    with zipfile.ZipFile(z_path, 'r') as z:
        img_names = [f for f in z.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) and not f.startswith('__MACOSX')]
        print(f"Total image files: {len(img_names)}")
        sizes = []
        resolutions = set()
        corrupt = 0
        for name in img_names:
            try:
                data = z.read(name)
                sizes.append(len(data))
                img = Image.open(io.BytesIO(data))
                resolutions.add((img.size, img.mode, img.format))
            except Exception as e:
                corrupt += 1
        print(f"File size (bytes): min={min(sizes) if sizes else 0}, mean={sum(sizes)//len(sizes) if sizes else 0}, max={max(sizes) if sizes else 0}")
        print(f"Corrupt count: {corrupt}")
        print(f"Unique resolutions/modes/formats (sample 5): {list(resolutions)[:5]}")
        total_valid_imgs += len(sizes) - corrupt

print(f"\nTotal valid images across all classes: {total_valid_imgs}")
