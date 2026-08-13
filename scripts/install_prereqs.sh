#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "Root dir: $$ROOT_DIR"

echo "Checking for system package manager..."
if command -v apt-get >/dev/null 2>&1; then
    echo "Using apt-get to install system packages (you may be prompted for sudo)..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip gnuplot
elif command -v yum >/dev/null 2>&1; then
    echo "Using yum to install system packages (you may be prompted for sudo)..."
    sudo yum install -y python3 python3-venv python3-pip gnuplot
else
    echo "No supported package manager detected. Please install: python3, python3-venv, python3-pip, gnuplot"
fi

VENV_DIR="$$ROOT_DIR/.venv"
if [ ! -d "$$VENV_DIR" ]; then
    echo "Creating virtualenv at $$VENV_DIR"
    python3 -m venv "$$VENV_DIR"
fi
echo "Activating virtualenv and installing Python packages..."
. "$$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$$ROOT_DIR/requirements.txt"

echo "Prerequisites installed. Activate the virtualenv with: source .venv/bin/activate"