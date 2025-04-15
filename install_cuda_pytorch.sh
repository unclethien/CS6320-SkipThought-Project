#!/bin/bash

# Update and upgrade packages
echo "Updating system packages..."
sudo apt update
sudo apt upgrade -y

# Install required dependencies
echo "Installing dependencies..."
sudo apt install -y build-essential gcc g++ make binutils
sudo apt install -y software-properties-common wget

# Install CUDA dependencies
echo "Installing CUDA dependencies..."
sudo apt install -y linux-headers-$(uname -r)

# Add NVIDIA repository
echo "Adding NVIDIA repository..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA Toolkit
echo "Installing CUDA Toolkit..."
sudo apt install -y cuda-toolkit-12-3

# Set up environment variables
echo 'export PATH=/usr/local/cuda-12.3/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc

# Install Python and pip if not already installed
echo "Installing Python and pip..."
sudo apt install -y python3 python3-pip python3-dev

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers library (needed for your script)
echo "Installing transformers library..."
pip3 install transformers

# Verify installations
echo "Verifying CUDA installation..."
nvcc --version

echo "Verifying PyTorch installation..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('Number of devices:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

echo "Installation complete!"
