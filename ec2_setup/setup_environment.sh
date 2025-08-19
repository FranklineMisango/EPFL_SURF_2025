#!/bin/bash
# setup_environment.sh - Install all required dependencies for the EPFL_SURF_2025 project

# Update system packages
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y build-essential python3-pip python3-dev git wget curl htop

echo "Installing NVIDIA drivers and CUDA..."
# Install NVIDIA drivers and CUDA
sudo apt-get install -y nvidia-driver-470 nvidia-cuda-toolkit
# Verify CUDA installation
nvcc --version
nvidia-smi

echo "Installing Python dependencies..."
# Create virtual environment
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
source venv/bin/activate

# Install base Python packages
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install jupyter ipython pandas numpy scikit-learn scipy plotly folium streamlit streamlit-folium matplotlib seaborn

# Install specialized packages
pip install geopy overpy requests networkx osmnx geopandas shapely pyogrio python-dateutil joblib tqdm xgboost lightgbm contextily rasterio fiona

# Install PyTorch Geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

echo "Environment setup complete!"
