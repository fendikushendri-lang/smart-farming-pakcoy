import zipfile
import hashlib
import os

zip_files = {
    "Masa Pertumbuhan": r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new\s3Masa Pertumbuhan.zip",
    "Mendekati Panen": r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new\s3Mendekati Panen pokcoy.zip",
    "Siap Panen": r"C:\Users\Asus\.gemini\antigravity\scratch\smart-farming-pakcoy-new\s3siap panen pokcoy.zip"
}

hashes = {}
duplicates = []

for label, z_path in zip_files.items():
    with zipfile.ZipFile(z_path, 'r') as z:
        for name in z.namelist():
            if name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')) and not name.startswith('__MACOSX'):
                data = z.read(name)
                md5 = hashlib.md5(data).hexdigest()
                if md5 in hashes:
                    duplicates.append((label, name, hashes[md5][0], hashes[md5][1]))
                else:
                    hashes[md5] = (label, name)

print(f"Total unique images: {len(hashes)}")
print(f"Total duplicate instances: {len(duplicates)}")
for dup in duplicates:
    print(f"Duplicate: [{dup[0]}] {os.path.basename(dup[1])} == [{dup[2]}] {os.path.basename(dup[3])}")
