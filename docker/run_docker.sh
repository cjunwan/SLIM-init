#!/usr/bin/env bash
# Start an interactive container with this repository mounted at /workspace/slim-init
# and a dataset directory mounted at /datasets.
#
#   ./docker/run_docker.sh [DATASET_DIR]
#
# DATASET_DIR defaults to $SLIM_INIT_DATASETS, then to $HOME/datasets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_DIR="${1:-${SLIM_INIT_DATASETS:-${HOME}/datasets}}"
IMAGE_TAG="${IMAGE_TAG:-slim-init:latest}"

if [ ! -d "${DATASET_DIR}" ]; then
    echo "Dataset directory not found: ${DATASET_DIR}" >&2
    exit 1
fi

# Allow the container to open OpenCV windows (show_track: 1 in the config).
xhost +local:docker >/dev/null 2>&1 || true

docker run -it --rm \
    --name slim-init \
    --net=host \
    --user "$(id -u):$(id -g)" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -e DISPLAY="${DISPLAY:-}" \
    -e QT_X11_NO_MITSHM=1 \
    -v "${PROJECT_DIR}":/workspace/slim-init \
    -v "${DATASET_DIR}":/datasets \
    -w /workspace/slim-init \
    "${IMAGE_TAG}"
