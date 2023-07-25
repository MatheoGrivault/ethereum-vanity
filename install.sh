#!/bin/bash

sudo apt-get update
sudo apt-get upgrade -y

gpu=$(lspci | grep -i nvidia)

if [ -n "$gpu" ]; then
    echo "NVIDIA Graphics Card Detected."
else
    echo "No NVIDIA Graphics Card Detected."
    exit 1
fi

echo "Checking CUDA dependencies..."
cuda=$(nvcc --version)
if [ $? -eq 0 ]; then
    echo "CUDA is already installed."
else
    echo "CUDA is not installed."
    echo "Starting installation..." 
    sudo apt-get install -y nvidia-cuda-toolkit
    sudo apt-get install -y nvidia-cuda-dev
    sudo apt-get install -y nvidia-cuda-doc
fi

echo "Checking for crypto dependencies..."
libsec=$(dpkg -l | grep libsecp256k1)
if [ -n "$libsec" ]; then
    echo "libsecp256k1 is already installed."
else
    echo "libsecp256k1 is not installed."
    echo "Starting installation..." 
    sudo apt-get install -y libsecp256k1-dev
fi

echo "Starting installation of ethereum-vanity software..."
wget https://github.com/MatheoGrivault/ethereum-vanity/releases/download/v1.0.0/ethereum-vanity-linux-v1.0.0.tar.gz
tar -xvf ethereum-vanity-linux-v1.0.0.tar.gz
rm ethereum-vanity-linux-v1.0.0.tar.gz
mv ethereum-vanity ~/Desktop/
chmod +x ~/Desktop/ethereum-vanity/ethereum-vanity

