# OpenMAPT1AutoParcellation.py
# ScriptedLoadableModule for 3D Slicer
import os
import qt
import ctk
import slicer
import vtk

# Otomatik paket kurulumu
def install_requirements():
    required = [
        "torch",
        "torchvision", 
        "numpy",
        "pandas",
        "nibabel",
        "gdown>=4.6.0",
        "openpyxl",
        "tqdm"
    ]
    import importlib
    for package in required:
        try:
            importlib.import_module(package)
        except ImportError:
            slicer.util.pip_install(package)

install_requirements()

import torch
import nibabel as nib
import numpy as np
import pandas as pd
import zipfile
import shutil

# Add module lib to path
import sys
module_dir = os.path.dirname(__file__)
lib_path = os.path.join(module_dir, "OpenMAPT1AutoParcellationLib")
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# Attempt to import OpenMAP utils
try:
    from utils.cropping import cropping
    from utils.hemisphere import hemisphere
    from utils.load_model import load_model
    from utils.parcellation import parcellation
    from utils.preprocessing import preprocessing
    from utils.stripping import stripping
    from utils.postprocessing import postprocessing
except Exception as e:
    utils_import_error = str(e)
else:
    utils_import_error = None

class OpenMAPT1AutoParcellation(slicer.ScriptedLoadableModule.ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "OpenMAP-T1 Auto Parcellation"
        parent.categories = ["Segmentation"]
        parent.contributors = ["Adapted for Slicer"]
        parent.helpText = "Run OpenMAP-T1 pipeline on a T1 MRI volume and load aligned labelmap."
        parent.acknowledgementText = "OpenMAP-T1"

class OpenMAPT1AutoParcellationWidget(slicer.ScriptedLoadableModule.ScriptedLoadableModuleWidget):

    GDRIVE_FILE_ID = "1YEE65X5Cx8LHK-C070TbZARHp1C08KxA"

    def setup(self):
        super().setup()
        self.moduleDir = os.path.dirname(__file__)
        self.layout = self.parent.layout()
        self.resultDataFrame = None
        self.outputFolder = None

        # Info
        info = qt.QLabel(
            "<b>OpenMAP-T1 Auto Parcellation</b><br>"
            "Select a T1 volume and press Run.<br><br>"
            "<b style='color: #d9534f;'>NON-COMMERCIAL USE ONLY</b><br>"
            "<small>Commercial use requires separate license.</small>"
        )
        info.setWordWrap(True)
        self.layout.addWidget(info)

        # --- MODEL DOWNLOAD ---
        downloadGroup = qt.QGroupBox("Model Download")
        downloadLayout = qt.QVBoxLayout()
        downloadInfo = qt.QLabel("<small>Models required (~1.5GB)</small>")
        downloadLayout.addWidget(downloadInfo)
        self.downloadButton = qt.QPushButton("Download Models from Google Drive")
        self.downloadButton.clicked.connect(self.onDownloadClicked)
        downloadLayout.addWidget(self.downloadButton)
        manualInfo = qt.QLabel(
            "<small><b>Manual:</b> <a href='https://drive.google.com/file/d/1YEE65X5Cx8LHK-C070TbZARHp1C08KxA/view?usp=sharing'>Google Drive Link</a></small>"
        )
        manualInfo.setOpenExternalLinks(True)
        downloadLayout.addWidget(manualInfo)
        downloadGroup.setLayout(downloadLayout)
        self.layout.addWidget(downloadGroup)

        # Volume selector
        self.layout.addWidget(qt.QLabel("Input T1 Volume:"))
        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.layout.addWidget(self.inputSelector)

        # Run button
        self.runButton = qt.QPushButton("Run OpenMAP-T1 Pipeline")
        self.runButton.enabled = True
        self.runButton.clicked.connect(self.onRunClicked)
        self.layout.addWidget(self.runButton)

        # --- EXCEL EXPORT ---
        exportGroup = qt.QGroupBox("Export Results")
        exportLayout = qt.QVBoxLayout()
        self.exportButton = qt.QPushButton("Export Volumes to Excel")
        self.exportButton.enabled = False
        self.exportButton.clicked.connect(self.onExportClicked)
        exportLayout.addWidget(self.exportButton)
        self.exportPathLabel = qt.QLabel("<small>Run pipeline first.</small>")
        self.exportPathLabel.setWordWrap(True)
        exportLayout.addWidget(self.exportPathLabel)
        exportGroup.setLayout(exportLayout)
        self.layout.addWidget(exportGroup)

        # Progress bar
        self.progressBar = qt.QProgressBar()
        self.progressBar.setVisible(False)
        self.layout.addWidget(self.progressBar)

        # Log
        self.layout.addWidget(qt.QLabel("Log:"))
        self.log = qt.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        self.layout.addWidget(self.log)
        self.layout.addStretch(1)

        self.checkModelsExist()

    def logMessage(self, text):
        self.log.append(text)
        slicer.app.processEvents()

    # ========== MODEL DOWNLOAD ==========

    def checkModelsExist(self):
        model_folder = os.path.join(self.moduleDir, "MODEL_FOLDER")
        if os.path.exists(model_folder) and os.listdir(model_folder):
            self.logMessage("Models found in MODEL_FOLDER")
            self.downloadButton.setText("Re-download Models")
            return True
        self.logMessage("Models not found. Please download first.")
        self.downloadButton.setText("Download Models (Required)")
        return False

    def onDownloadClicked(self):
        msgBox = qt.QMessageBox()
        msgBox.setWindowTitle("License Agreement")
        msgBox.setIcon(qt.QMessageBox.Information)
        msgBox.setText(
            "<b>NON-COMMERCIAL USE ONLY</b><br><br>"
            "By downloading you agree to non-commercial use only.<br>"
            "<a href='https://github.com/OishiLab/OpenMAP-T1/blob/main/LICENSE'>Full License</a><br><br>"
            "<b>Do you accept? (~1.5GB)</b>"
        )
        msgBox.setStandardButtons(qt.QMessageBox.Yes | qt.QMessageBox.No)
        if msgBox.exec_() == qt.QMessageBox.Yes:
            self.downloadModels()

    def downloadModels(self):
        model_folder = os.path.join(self.moduleDir, "MODEL_FOLDER")
        if os.path.exists(model_folder):
            shutil.rmtree(model_folder)
        try:
            slicer.util.pip_install("gdown --upgrade")
            import importlib
            import gdown
            importlib.reload(gdown)

            self.progressBar.setVisible(True)
            self.progressBar.setRange(0, 0)
            slicer.app.processEvents()

            temp_zip = os.path.join(self.moduleDir, "models_temp.zip")
            url = "https://drive.google.com/uc?id=" + self.GDRIVE_FILE_ID

            self.logMessage("Downloading models...")
            gdown.download(url, temp_zip, quiet=False)

            self.logMessage("Extracting...")
            with zipfile.ZipFile(temp_zip, 'r') as z:
                z.extractall(self.moduleDir)
                # Fix Windows-style backslash paths for Mac/Linux compatibility
            for root, dirs, files in os.walk(self.moduleDir):
                for filename in files:
                if '\\' in filename:
                    old_path = os.path.join(root, filename)
                    parts = filename.split('\\')
                    new_dir = os.path.join(root, *parts[:-1])
                    os.makedirs(new_dir, exist_ok=True)
                    new_path = os.path.join(new_dir, parts[-1])
                    shutil.move(old_path, new_path)
            if os.path.exists(temp_zip):
                os.remove(temp_zip)

            self.progressBar.setVisible(False)
            self.logMessage("Models downloaded!")
            self.checkModelsExist()

        except Exception as e:
            self.progressBar.setVisible(False)
            self.logMessage("ERROR: " + str(e))
            import traceback
            self.logMessage(traceback.format_exc())

    # ========== EXCEL EXPORT ==========

    def onExportClicked(self):
        if self.resultDataFrame is None:
            self.logMessage("No results. Run pipeline first.")
            return
        try:
            excel_path = os.path.join(self.outputFolder, "T1_280_volumes.xlsx")
            self.resultDataFrame.to_excel(excel_path, index=False, sheet_name="Brain_Volumes")
            self.logMessage("Excel saved: " + excel_path)
            self.exportPathLabel.setText("<small><b>Saved:</b> " + excel_path + "</small>")
            import subprocess
            subprocess.Popen(["explorer", self.outputFolder])
        except Exception as e:
            self.logMessage("Export error: " + str(e))

    # ========== LOAD LABELS FROM labeled.txt ==========

    def loadLabels(self):
        label_dict = {}
        labeled_path = os.path.join(self.moduleDir, "labeled.txt")
        if not os.path.exists(labeled_path):
            self.logMessage("labeled.txt not found: " + labeled_path)
            return label_dict
        try:
            with open(labeled_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    try:
                        idx = int(parts[0])
                        name = ' '.join(parts[7:]).strip('"')
                        if name and idx != 0:
                            label_dict[idx] = name
                    except:
                        pass
            self.logMessage("Loaded " + str(len(label_dict)) + " labels from labeled.txt")
        except Exception as e:
            self.logMessage("labeled.txt error: " + str(e))
        return label_dict

    # ========== RUN PIPELINE ==========

    def onRunClicked(self):
        try:
            model_folder = os.path.join(self.moduleDir, "MODEL_FOLDER")
            if not os.path.exists(model_folder) or not os.listdir(model_folder):
                qt.QMessageBox.warning(None, "Error", "Models not found! Download first.")
                return
            self.runPipeline()
        except Exception as e:
            self.logMessage("ERROR: " + str(e))
            import traceback
            self.logMessage(traceback.format_exc())

    def runPipeline(self):
        if utils_import_error:
            raise RuntimeError("utils import failed: " + utils_import_error)

        os.chdir(self.moduleDir)

        volumeNode = self.inputSelector.currentNode()
        if not volumeNode:
            raise RuntimeError("No T1 volume selected.")

        self.logMessage("Starting pipeline...")

        output_folder = os.path.join(self.moduleDir, "output")
        os.makedirs(output_folder, exist_ok=True)
        self.outputFolder = output_folder

        tmp_t1_path = os.path.join(output_folder, "T1_tmp.nii.gz")
        slicer.util.saveNode(volumeNode, tmp_t1_path)
        self.logMessage("T1 saved.")

        # Load models
        model_folder = os.path.join(self.moduleDir, "MODEL_FOLDER")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logMessage("Loading models...")

        class Opt:
            m = model_folder

        cnet, ssnet, pnet_c, pnet_s, pnet_a, hnet_c, hnet_a = load_model(Opt(), device)
        self.logMessage("Models loaded.")

        # Pipeline
        self.logMessage("Preprocessing...")
        odata, data = preprocessing(tmp_t1_path, output_folder, "T1")
        self.logMessage("Cropping...")
        cropped = cropping(data, cnet, device)
        self.logMessage("Stripping...")
        stripped, shift = stripping(cropped, data, ssnet, device)
        self.logMessage("Parcellating...")
        parcellated = parcellation(stripped, pnet_c, pnet_s, pnet_a, device)
        self.logMessage("Hemisphere...")
        separated = hemisphere(stripped, hnet_c, hnet_a, device)
        self.logMessage("Postprocessing...")
        aligned_output = postprocessing(parcellated, separated, shift, device)

        # Load labels from labeled.txt
        label_dict = self.loadLabels()

        # Calculate volumes
        self.logMessage("Calculating volumes...")
        voxel_volume = np.prod(data.header.get_zooms())
        unique_labels, counts = np.unique(aligned_output, return_counts=True)
        mask = unique_labels != 0
        unique_labels = unique_labels[mask]
        counts = counts[mask]
        volumes_mm3 = counts * voxel_volume

        df = pd.DataFrame({
            "LabelID": unique_labels.astype(int),
            "Volume_mm3": volumes_mm3
        })
        if label_dict:
            df["LabelName"] = df["LabelID"].map(label_dict).fillna("")
        else:
            df["LabelName"] = ""

        # Save CSV
        csv_path = os.path.join(output_folder, "T1_280_volumes.csv")
        df.to_csv(csv_path, index=False)
        self.logMessage("CSV saved.")

        # Store for export
        self.resultDataFrame = df
        self.exportButton.setEnabled(True)
        self.exportPathLabel.setText("<small>Results ready. Click Export.</small>")

        # Auto-save Excel too
        try:
            excel_path = os.path.join(output_folder, "T1_280_volumes.xlsx")
            df.to_excel(excel_path, index=False, sheet_name="Brain_Volumes")
            self.logMessage("Excel saved.")
        except Exception as e:
            self.logMessage("Excel auto-save failed: " + str(e))

        # Save labelmap
        nii = nib.Nifti1Image(aligned_output.astype(np.uint16), affine=data.affine)
        out_label = os.path.join(output_folder, "T1_280_segment.nii.gz")
        nib.save(nii, out_label)
        self.logMessage("Labelmap saved.")

        # Load into Slicer 2D
        labelNode = slicer.util.loadLabelVolume(out_label)
        labelNode.SetName("OpenMAP_T1_Labelmap")
        labelNode.SetAndObserveTransformNodeID(None)
        labelNode.GetDisplayNode().SetOpacity(0.4)
        slicer.util.setSliceViewerLayers(background=volumeNode, foreground=labelNode, foregroundOpacity=0.4)
        self.logMessage("2D labelmap loaded.")

        # 3D Segmentation
        try:
            self.logMessage("Creating 3D segmentation...")
            segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            segNode.SetName("OpenMAP_T1_Segmentation")
            segLogic = slicer.modules.segmentations.logic()
            segLogic.ImportLabelmapToSegmentationNode(labelNode, segNode)
            segNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
            segNode.CreateClosedSurfaceRepresentation()

            # Rename segments from labeled.txt
            if label_dict:
                segmentation = segNode.GetSegmentation()
                segmentIds = vtk.vtkStringArray()
                segmentation.GetSegmentIDs(segmentIds)
                renamed = 0
                for i in range(segmentIds.GetNumberOfValues()):
                    segId = segmentIds.GetValue(i)
                    segment = segmentation.GetSegment(segId)
                    if segment:
                        try:
                            parts = segment.GetName().split("_")
                            label_id = int(parts[-1])
                            if label_id in label_dict:
                                segment.SetName(label_dict[label_id])
                                renamed += 1
                        except:
                            pass
                self.logMessage("Renamed " + str(renamed) + " segments")

            displayNode = segNode.GetDisplayNode()
            if displayNode:
                displayNode.SetVisibility3D(True)
                displayNode.SetOpacity3D(0.6)
                displayNode.SetVisibility2DFill(True)
                displayNode.SetVisibility2DOutline(True)

            layoutManager = slicer.app.layoutManager()
            if layoutManager:
                threeDWidget = layoutManager.threeDWidget(0)
                if threeDWidget:
                    threeDView = threeDWidget.threeDView()
                    threeDView.resetFocalPoint()
                    threeDView.resetCamera()

            self.logMessage("3D segmentation done!")

        except Exception as e:
            self.logMessage("3D seg error: " + str(e))
            import traceback
            self.logMessage(traceback.format_exc())

        self.logMessage("=" * 50)
        self.logMessage("PIPELINE COMPLETED")
        self.logMessage("Regions: " + str(len(unique_labels)))
        self.logMessage("Output: " + output_folder)
        self.logMessage("=" * 50)
