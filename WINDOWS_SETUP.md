# Windows Setup Guide for DepthFusion-ViT Development

This guide will help you set up your Windows development environment for training and testing the DepthFusion-ViT model.

## Prerequisites

1. **Python 3.10** (Required for TensorFlow 2.10 with GPU support on Windows)
2. **NVIDIA GPU** with CUDA support
3. **NVIDIA Graphics Driver** (Latest version recommended)
4. **Git** for Windows

## Step 1: Install Python 3.10

1. Download Python 3.10 from the official website:
   https://www.python.org/downloads/release/python-3109/
   
2. During installation:
   - Check "Add Python 3.10 to PATH"
   - Choose "Customize installation"
   - Enable all optional features
   - Set installation path to: `C:\Python310`

## Step 2: Install NVIDIA CUDA Toolkit 11.2

1. Download CUDA Toolkit 11.2 from NVIDIA:
   https://developer.nvidia.com/cuda-11.2.0-download-archive
   
2. Run the installer and follow these steps:
   - Choose "Custom" installation
   - Select all components
   - Install

## Step 3: Install cuDNN 8.1.0

1. Download cuDNN 8.1.0 for CUDA 11.2 from NVIDIA (requires free account):
   https://developer.nvidia.com/cudnn
   
2. Extract the downloaded archive
3. Copy files to your CUDA installation:
   - Copy `cuda\bin\*` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
   - Copy `cuda\include\*` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include`
   - Copy `cuda\lib\x64\*` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64`

## Step 4: Set Up Python Virtual Environment

```bash
# Create project directory
mkdir E:\projects\depthfusion-vit
cd E:\projects\depthfusion-vit

# Create virtual environment with Python 3.10
python -m venv venv --python=3.10

# Activate virtual environment
venv\Scripts\activate

# Verify Python version
python --version
# Should output: Python 3.10.x

# Upgrade pip
python -m pip install --upgrade pip
```

### 1. Wrong Python Version

If you see a different Python version after activating the virtual environment:
```bash
# Remove existing venv
rmdir /s /q venv

# Create new venv with Python 3.10
py -3.10 -m venv venv

# Activate and verify
venv\Scripts\activate
python --version
```

## Step 5: Install Required Packages

With the virtual environment activated:

```bash
# Install TensorFlow and CUDA packages
pip install tensorflow==2.10.0

# Install other requirements
pip install -r requirements.txt
```

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install facenet-pytorch --no-deps

## Step 6: Verify Installation

Run the test script to verify your setup:

```bash
python test_gpu.py
```

You should see successful tests for TensorFlow GPU, PyTorch GPU, and OpenCV.

## Common Issues and Solutions

### 1. CUDA Not Found

If TensorFlow can't find CUDA, verify your environment variables:

```bash
echo %PATH%
```

Should include:
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
- `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp`

### 2. TensorFlow Import Error

If you get an error importing TensorFlow, try:

```bash
pip uninstall tensorflow tensorflow-gpu
pip install tensorflow==2.10.0
```

### 3. CUDA Out of Memory

If you get CUDA out of memory errors during training:
- Reduce batch size in training configuration
- Close other GPU-intensive applications
- Update NVIDIA drivers

## Next Steps

1. Run data collection:
   ```bash
   python data_collection_code.py
   ```

2. Train the model:
   ```bash
   python train_model.py --data_dir=./data --output_dir=./models
   ```

## Project Structure

```
depthfusion-vit/
├── data/
│   ├── real/
│   └── fake/
├── models/
├── venv/
├── data_collection_code.py
├── train_model.py
├── test_gpu.py
├── requirements.txt
└── WINDOWS_SETUP.md
```

## Notes

- The model is designed to be trained on Windows with CUDA and deployed on both Windows and Raspberry Pi
- For Raspberry Pi deployment, we'll convert the trained model to TensorFlow Lite format
- Keep your NVIDIA drivers updated for best performance
- Monitor GPU temperature during long training sessions 