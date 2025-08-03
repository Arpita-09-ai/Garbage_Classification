# Garbage Classification - Installation Guide

## Prerequisites
- **Python 3.8+** (3.9-3.11 recommended)
- **8GB+ RAM** (16GB recommended for training)
- **5GB+ storage** for datasets and models
- **Git** and **pip** installed

Verify installation:
```bash
python --version  # Should show 3.8+
pip --version
```

## Quick Setup

```bash
# 1. Fork repository (if contributing)
# Go to https://github.com/AditixAnand/Garbage_Classification.git
# Click "Fork" button to create your own copy

# 2. Clone repository (use your fork URL if contributing)
git clone https://github.com/YOUR_USERNAME/Garbage_Classification.git
# Or clone original:
# git clone https://github.com/AditixAnand/Garbage_Classification.git
cd Garbage_Classification

# 3. Create virtual environment
python -m venv .venv

# 4. Activate environment
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Start Jupyter
jupyter notebook
```

## Dataset Setup

**Automatic (Recommended)**: Run the KaggleHub download cell in any notebook - dataset downloads automatically to `~/.cache/kagglehub/`

**Manual**: If automatic fails, organize your dataset as:
```
TrashType_Image_Dataset/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

## Running Notebooks

**Execution Order**:
1. **Week_1.ipynb** - Data exploration and basic preprocessing
2. **Week_2.ipynb** - EfficientNetV2B2 model training + Gradio interface  
3. **Week_3.ipynb** - Advanced training and deployment

**Alternative to Jupyter**: Use VS Code with Python extension for notebook editing.

## Key Dependencies

```
tensorflow>=2.13.0      # Deep learning framework
gradio==4.44.0          # Web interface
scikit-learn>=1.3.0     # ML utilities
matplotlib>=3.7.0       # Visualization
kagglehub>=0.2.0        # Dataset downloads
```

## Model Architecture

- **Base**: EfficientNetV2B2 with ImageNet weights
- **Input**: 124×124×3 RGB images
- **Classes**: 6 (cardboard, glass, metal, paper, plastic, trash)
- **Target Accuracy**: >95%
- **Deployment**: Gradio web interface

## Troubleshooting

**Common Issues**:

```bash
# TensorFlow GPU (M1/M2 Mac)
pip install tensorflow-macos tensorflow-metal

# Memory errors - reduce batch size in notebooks
batch_size = 16  # instead of 32

# Package conflicts
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Virtual environment issues (Windows)
.venv\Scripts\activate.bat
```

**Port Conflicts**: Notebooks auto-find available ports. Restart kernel if needed.

## Project Structure

```
Garbage_Classification/
├── Week_1.ipynb         # Data exploration
├── Week_2.ipynb         # EfficientNet training
├── Week_3.ipynb         # Advanced deployment
├── requirements.txt     # Dependencies
├── INSTALL.md          # This guide
└── .venv/              # Virtual environment
```

## Development

**Environment Setup**:
```bash
# Always activate before working
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Verify TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"
```

**Contributing**: Fork → Create branch → Follow setup → Make changes → Submit PR

---

**Need Help?** Create an issue on GitHub with your error details and system info.