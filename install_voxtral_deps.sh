#!/usr/bin/env bash
set -euo pipefail

echo "=== Update package list ==="
sudo apt-get update -y

echo "=== Install Python 3, pip, and ffmpeg ==="
sudo apt-get install -y python3 python3-pip ffmpeg

echo "=== Upgrade pip ==="
python3 -m pip install --upgrade pip

echo "=== Install core Python libraries ==="
# torch CPU version by default
python3 -m pip install --upgrade \
  torch torchvision torchaudio \
  transformers accelerate soundfile ffmpeg-python

echo "=== Installation complete ==="
echo "You can now run: python3 voxtral_subs.py input.mp4 --out captions.srt"

echo
echo "=== Optional: Install GPU-accelerated PyTorch ==="
echo "If you have an NVIDIA GPU with CUDA, install a matching wheel:"
echo "  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio"
