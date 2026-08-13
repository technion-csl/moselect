#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "Root dir: $ROOT_DIR"

echo "Checking for system package manager..."
if command -v apt-get >/dev/null 2>&1; then
    echo "Using apt-get to install system packages (you may be prompted for sudo)..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip
elif command -v yum >/dev/null 2>&1; then
    echo "Using yum to install system packages (you may be prompted for sudo)..."
    sudo yum install -y python3 python3-venv python3-pip
else
    echo "No supported package manager detected. Please install: python3, python3-venv, python3-pip"
fi

VENV_DIR="$ROOT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
echo "Activating virtualenv and installing Python packages..."
. "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$ROOT_DIR/requirements.txt"

echo "Installing perf (performance events) tooling..."
if command -v perf >/dev/null 2>&1 && perf --version >/dev/null 2>&1; then
    echo "perf is already installed: $(perf --version)"
else
    KERNEL_VERSION=$(uname -r)
    PERF_PACKAGE="linux-tools-$KERNEL_VERSION"
    if sudo apt-get install -y "$PERF_PACKAGE" 2>/dev/null; then
        echo "Successfully installed $PERF_PACKAGE"
    else
        echo "Could not find a linux-tools package matching kernel '$KERNEL_VERSION' (this is expected on WSL2 and other custom-built kernels)."
        echo "Attempting to install generic linux-tools packages..."
        if sudo apt-get install -y linux-tools-generic linux-tools-common 2>/dev/null; then
            echo "Successfully installed linux-tools-generic and linux-tools-common"
        else
            echo "WARNING: failed to install a working 'perf' binary for kernel '$KERNEL_VERSION'."
            echo "Please install 'perf' manually (matching or compatible with your kernel) and ensure it is on PATH."
            echo "You can continue using the codebase, but benchmark performance measurements may not work."
        fi
    fi
fi

echo "Prerequisites installed. Activate the virtualenv with: source .venv/bin/activate"