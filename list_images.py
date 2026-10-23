import zipfile
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

zip_files = {
    "Masa Pertumbuhan": r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new\s3Masa Pertumbuhan.zip",
    "Mendekati Panen": r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new\s3Mendekati Panen pokcoy.zip",
    "Siap Panen": r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new\s3siap panen pokcoy.zip"
}

for label, z_path in zip_files.items():
    print(f"\n==============================")
    print(f"CLASS: {label}")
    print(f"==============================")
    with zipfile.ZipFile(z_path, 'r') as z:
        img_names = [f for f in z.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) and not f.startswith('__MACOSX')]
        print(f"Count: {len(img_names)}")
        for i, name in enumerate(img_names):
            base = os.path.basename(name)
            info = z.getinfo(name)
            print(f"  {i+1:2d}. {base} ({info.file_size:,} bytes)")
