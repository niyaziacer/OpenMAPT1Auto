import os
import sys

try:
    import gdown
except ImportError:
    print("gdown not found. Install it via 'pip install gdown'")
    sys.exit(1)

# MODEL_FOLDER klasörü: komut satırından alınabilir, yoksa default
if len(sys.argv) > 1:
    model_folder = sys.argv[1]
else:
    model_folder = "MODEL_FOLDER"  # default path

os.makedirs(model_folder, exist_ok=True)

# Google Drive file ID’leri (örnek ID’ler, kendi Drive linkine göre değiştirin)
files = {
    "CNet.pth": "GDRIVE_FILE_ID_1",
    "HNet_axial.pth": "GDRIVE_FILE_ID_2",
    "HNet_coronal.pth": "GDRIVE_FILE_ID_3",
    "PNet_axial.pth": "GDRIVE_FILE_ID_4",
    "PNet_coronal.pth": "GDRIVE_FILE_ID_5",
    "PNet_sagittal.pth": "GDRIVE_FILE_ID_6",
    "SSNet.pth": "GDRIVE_FILE_ID_7",
    "penMAP-T1.pth": "GDRIVE_FILE_ID_8"
}

for fname, file_id in files.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    out_path = os.path.join(model_folder, fname)
    if not os.path.exists(out_path):
        print(f"Downloading {fname}...")
        gdown.download(url, out_path, quiet=False)
    else:
        print(f"{fname} already exists, skipping.")
