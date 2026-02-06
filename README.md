# isMap (immunological synapse map analysis program)

This repository provides a **isMap-Napari** plugin for analyzing T-cell activation from microscopy images.  
It integrates **data preprocessing, segmentation, feature extraction, and interactive visualization** in a single workflow.

---

## ✨ Features

- **Microscopy format support**
- **Background correction** (e.g. rolling-ball, Gaussian, top-hat)
- **T-cell segmentation** with [Cellpose](https://www.cellpose.org/) (including denoising models)
- **Per-cell feature extraction** (intensity metrics, shape descriptors, etc.)
- **Single-cell cropping**
- **Interactive visualization in Napari**:
  - Image + mask overlay  
  - Cell labels and per-cell properties
  - Feature-based filtering
  - Radial profiles and condition-wise comparisons
  - Export of results as CSV and images 

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/.../isMap-napari.git
cd isMap-napari
```

### 2. Create and activate a virtual environment
Conda is strongly recommended.
```bash
conda create -n venv-ismap -c conda-forge python=3.11
conda activate venv-ismap
```
ℹ️ The plugin is tested on Python 3.11.

### 3. Install the plugin
From inside the repo:
```bash
pip install -e ".[gui]"
```

### 4. Validate installation
```bash
npe2 validate ismap-napari
```
You should see:
```
✔ Manifest for 'immunological synapse Map analysis program' valid!
```

---

## 🧪 Usage in Napari

1. Launch Napari:
   ```bash
   napari
   ```
2. Open the plugin:
   **Plugins → isMap (immunological synapse Map analysis program)**
3. In the docked widget:
   - **Select an Input Folder containing microscopy images**
   - **Select an Output Folder for results**
   - **Choose segmentation parameters (model, diameter, scale)**
   - **Run segmentation**
   - **Run full analysis**
4. Visualization features include:
   - Multi-channel image layers
   - Segmentation masks and cell outlines
   - Cell-level labels and properties
   - Interactive filtering by morphology and intensity
   - Radial profiles and condition comparisons
   - Export of filtered results to CSV

---

## 📦 Requirements

- Python ≥ 3.10
- Napari + Qt (via conda)
- Cellpose + PyTorch
- Scientific Python stack (numpy, pandas, scikit-image, etc.)

All dependencies are resolved automatically when installing via conda + pip.

---
## Notes on GPU Support

Cellpose uses **PyTorch**
If a CUDA-compatible NVIDIA GPU is available, Cellpose will run on GPU automatically.
On macOS, Cellpose typically runs on CPU.

---

## 📜 License
BSD-3-Clause  


