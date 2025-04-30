#!/usr/bin/env bash
set -euo pipefail

echo "Detecting platform…"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  echo "→ Linux detected: installing via apt"
  sudo apt-get update
  sudo apt-get install -y \
    build-essential cmake pkg-config \
    libglfw3-dev libgl1-mesa-dev libomp-dev

elif [[ "$OSTYPE" == "darwin"* ]]; then
  echo "→ macOS detected: installing via Homebrew"
  if ! command -v brew >/dev/null; then
    echo "Homebrew not found! Install it here: https://brew.sh"
    exit 1
  fi
  brew update
  brew install cmake pkg-config glfw libomp

elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* || "$OS" == "Windows_NT" ]]; then
  echo "→ Windows detected: installing via vcpkg"
  # Clone vcpkg if needed
  if [ ! -d "vcpkg" ]; then
    git clone https://github.com/microsoft/vcpkg.git
    cd vcpkg
    ./bootstrap-vcpkg.sh
    cd ..
  fi
  # Install dependencies in the default triplet (adjust if you are x86)
  ./vcpkg/vcpkg install \
    glfw3

  echo "When you run CMake, add:"
  echo "  -DCMAKE_TOOLCHAIN_FILE=\$(pwd)/vcpkg/scripts/buildsystems/vcpkg.cmake"

else
  echo "Unsupported OS: $OSTYPE"
  exit 1
fi

echo "System dependencies installed!"

