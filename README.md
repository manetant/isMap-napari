# isMap (immunological synapse map analysis program)

This repository provides a **Napari plugin** for analyzing T-cell activation from microscopy images.  
It integrates **data preprocessing, segmentation, feature extraction, and interactive visualization** in a single workflow.

---

## ✨ Features

- **ND2 → TIFF conversion**
- **Background correction** (e.g. rolling ball for ICAM1 channel)
- **T-cell segmentation** with [Cellpose](https://www.cellpose.org/) (including denoising models)
- **Per-cell feature extraction** (intensity metrics, shape, circularity, etc.)
- **Single-cell image cropping**
- **Interactive visualization in Napari**:
  - Image + mask overlay  
  - Per-cell properties and text labels  
  - Filtering by shape features  
  - Export filtered results as CSV  

---

## 📂 Project Structure

```bash
predict-tcell/         # Python package (Napari plugin)
├── src/predict_tcell  # Source code
│   ├── analysis.py    # Main analysis pipeline
│   ├── preprocessing/ # Background correction etc.
│   ├── masking/       # Segmentation (Cellpose)
│   ├── metrics.py     # Per-cell features
│   ├── visualization/ # Napari visualization
│   ├── _widget.py     # Napari widget definition
│   └── napari.yaml    # Plugin manifest
└── test_data/         # Example input files
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/.../predict_tcell_plugin.git
cd predict_tcell_plugin/predict-tcell
```

### 2. Create and activate a virtual environment
Linux/macOS:
```bash
python3 -m venv .venv-predict
source .venv-predict/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv .venv-predict
.venv-predict\Scripts\Activate.ps1
```

### 3. Install the plugin
From inside the repo:
```bash
pip install -e .[all]
```

### 4. Validate installation
```bash
npe2 validate predict-tcell
```
You should see:
```
✔ Manifest for 'Predict T Cell' valid!
```

---

## 🧪 Usage in Napari

1. Start Napari:
   ```bash
   napari
   ```
2. Open the plugin:
   **Plugins → isMap (immunological synapse map analysis program)**
3. In the docked widget:
   - **Input Folder** → folder with `.nd2` files  
   - **Output Folder** → where results are saved  
   - **Channels** → e.g. `ICAM1,pTyr,Actin`  
   - **Run Analysis** → runs processing with progress bar  
4. After processing, results appear in the same Napari window:
   - Multi-channel images  
   - Actin segmentation masks  
   - Points layer with per-cell properties + text labels  
   - Interactive filters (circularity, eccentricity, diameter)  
   - CSV export widget (choose save location)  

---

## 📦 Requirements

- Python **3.10+**
- Core:
  - `numpy`, `pandas`, `scikit-image`, `opencv-python`, `tifffile`
- Deep learning:
  - `torch`, `torchvision`, `cellpose==3.1.1.2`
- Napari & GUI:
  - `napari[all]`, `magicgui`, `qtpy`
- Others:
  - `scikit-learn`, `nd2reader`

---

## 📜 License
BSD-3-Clause  

