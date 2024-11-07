#!/bin/bash

# Update package list
sudo apt-get update

# Install system dependencies
sudo apt-get install -y \
    python3-distutils \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgtk2.0-dev \
    tesseract-ocr

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install setuptools
pip install --upgrade pip setuptools wheel

# Install PyTorch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt