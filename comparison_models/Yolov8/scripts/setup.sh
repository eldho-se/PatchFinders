#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- Configuration ---
ENV_NAME="patch_finder"
PYTHON_VERSION="3.10"

echo "=== Starting Patch Finder Environment Setup ==="

# 1. Check for Conda and install if missing
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Installing Miniconda..."
    
    # Download the installer based on OS (Fixed URLs)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o miniconda.sh
    else
        echo "Unsupported OS for automated install. Please install Conda manually."
        exit 1
    fi

    # Install silently
    bash miniconda.sh -b -p "$HOME/miniconda"
    rm miniconda.sh

    # Initialize Conda for the current shell session
    eval "$("$HOME/miniconda/bin/conda" shell.bash hook)"
    conda init
    echo "Miniconda installed successfully."
else
    echo "Conda is already installed."
    # Ensure conda commands are available in this script
    eval "$(conda shell.bash hook)"
fi

# 2. Create the environment
if [ -f "environment.yml" ]; then
    echo "Found environment.yml. Building environment..."
    # Using 'conda env update' allows the script to be run multiple times safely
    conda env update -f environment.yml --prune
else
    echo "No environment.yml found. Creating empty environment..."
    conda create --name $ENV_NAME python=$PYTHON_VERSION -y
fi

echo "======================================"
echo "Setup Complete!"
echo "To start working on Patch Finder, run:"
echo "conda activate $ENV_NAME"
echo "======================================"