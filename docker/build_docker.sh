#!/usr/bin/env bash
# Build the SLIM-init development image (Ubuntu 22.04 + OpenCV 4.12 with contrib + Ceres).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="${IMAGE_TAG:-slim-init:latest}"

echo ">>> Building ${IMAGE_TAG} ..."
docker build -t "${IMAGE_TAG}" "${SCRIPT_DIR}"
