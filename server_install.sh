#!/bin/bash

# Setting variables
REPO_URL="https://github.com/asigatchov/vball-net.git"
BRANCH="player-net"
PROJECT_DIR="/home/vball-net"
DATASET_URL="https://volleyball-orel.ru/system/docs/data_20250711_2330.tgz"
DATASET_DIR="/home/vball-net/data"

# Checking SSH connection (assuming SSH is already configured)
echo "Connecting to the server..."

# Installing git
echo "Installing git..."
sudo apt-get update
sudo apt-get install -y git

# Cloning the repository
echo "Cloning the repository..."
if [ -d "$PROJECT_DIR" ]; then
    echo "Directory $PROJECT_DIR already exists, removing..."
    rm -rf "$PROJECT_DIR"
fi
git clone --branch "$BRANCH" "$REPO_URL" "$PROJECT_DIR"

# Installing uv
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Changing to the project directory
echo "Changing to the project directory: $PROJECT_DIR"
cd "$PROJECT_DIR" || exit 1

# Running uv sync
echo "Running uv sync..."
uv sync

# Downloading and unpacking the dataset
echo "Downloading the dataset..."
mkdir -p "$DATASET_DIR"
curl -L "$DATASET_URL" -o dataset.tgz
echo "Unpacking the dataset to $DATASET_DIR..."
tar -xzf dataset.tgz -C "$PROJECT_DIR"
rm dataset.tgz

# Running preprocess.py
echo "Running preprocess.py..."
uv run src/preprocess.py

# Starting training
echo "Starting model training..."
uv run src/train_v1.py --grayscale --seq 9 --model_name VballNetFastV1 --epochs 350 --resume --gpu_memory_limit 2600 --alpha 0.5

echo "Deployment completed!"
