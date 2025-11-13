# OpenMAPT1Auto

**OpenMAPT1Auto** is a 3D Slicer module that automatically parcellates the human brain into **280 anatomical regions** from T1-weighted MRI scans and computes their volumes.

---

## Features

- Fully automated brain parcellation into 280 regions.
- Calculates volumes for each labeled structure.
- Compatible with 3D Slicer (v5.x recommended).
- Outputs:
  - NIfTI segmented volume
  - CSV file with LabelID and Volume in mm³
  - Optional color-coded label map visualization
  - Label lookup file for ITK-SNAP (`Untitled.txt`)

---

## Installation

1. Clone the repository or download it as a ZIP:

```bash
git clone https://github.com/niyaziacer/OpenMAPT1Auto.git


Open 3D Slicer.

Go to Edit → Application Settings → Modules and add the path to the module folder (Windows example):



Restart 3D Slicer. The module should appear in the Modules list.

Usage

Load a T1-weighted brain MRI (NIfTI format) in 3D Slicer.

Open the OpenMAPT1Auto module.

Specify:

Input folder: folder containing your MRI(s)

Output folder: folder where results will be saved

Model folder: folder containing pretrained model weights

Click Run.

Example Data

MRHead_1.nii.gz → Example T1-weighted MRI for testing

T1_280_segment.nii.gz → Example parcellated brain volume

T1_280_volumes.csv → Example CSV with volumes

Untitled.txt → Label lookup file for ITK-SNAP

Pretrained Models

Note: Pretrained model weights are not included in this repository due to license restrictions (JHU Research Software License Agreement – No For-Profit, No Redistribution).

Manual download

Download the models from Google Drive:

OpenMAP-T1 Models

Place all model files (including penMAP-T1.pth) into your MODEL_FOLDER before running the module.

Automatic download via Python

You can also use this Python snippet to download the models:

import os
import gdown

model_folder = "MODEL_FOLDER"  # Replace with your path
os.makedirs(model_folder, exist_ok=True)

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



Make sure gdown is installed:


pip install gdown


Visualization

Load T1_280_segment.nii.gz in 3D Slicer and apply a color table (e.g., FullRainbow) if desired.

Below is an example of the parcellated brain:

Each color corresponds to a different anatomical region, and volumes are calculated automatically.




License

This project is licensed under the MIT License. See LICENSE for details.

Important: Pretrained models are subject to the JHU Research Software License Agreement – No For-Profit, No Redistribution.
This software and models are intended for educational and research purposes only, not for any commercial or profit-driven use.


