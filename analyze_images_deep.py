import zipfile
import io
from PIL import Image
import pandas as pd

zip_files = {
    "Masa Pertumbuhan": r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new\s3Masa Pertumbuhan.zip",
    "Mendekati Panen": r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new\s3Mendekati Panen pokcoy.zip",
    "Siap Panen": r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new\s3siap panen pokcoy.zip"
}

data = []
for label, z_path in zip_files.items():
    with zipfile.ZipFile(z_path, 'r') as z:
        for name in z.namelist():
            if name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) and not name.startswith('__MACOSX'):
                raw = z.read(name)
                img = Image.open(io.BytesIO(raw))
                w, h = img.size
                data.append({
                    'class': label,
                    'filename': name,
                    'width': w,
                    'height': h,
                    'aspect_ratio': round(w / h, 2),
                    'mode': img.mode,
                    'format': img.format,
                    'size_kb': round(len(raw) / 1024, 1)
                })

img_df = pd.DataFrame(data)
print("=== IMAGE DATASET SUMMARY ===")
print("Total rows:", len(img_df))
print(img_df.groupby('class').agg({
    'filename': 'count',
    'width': ['min', 'mean', 'max'],
    'height': ['min', 'mean', 'max'],
    'size_kb': ['min', 'mean', 'max']
}))

print("\nColor modes:", img_df['mode'].value_counts())
print("Formats:", img_df['format'].value_counts())

print("\nSample low-resolution images (< 300px):")
low_res = img_df[(img_df['width'] < 300) | (img_df['height'] < 300)]
print(f"Total low-res: {len(low_res)}")
print(low_res[['class', 'filename', 'width', 'height', 'size_kb']].head(10))
