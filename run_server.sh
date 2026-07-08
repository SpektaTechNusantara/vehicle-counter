#!/bin/bash
# Launches the multi-camera vehicle counter web server.
#
# Usage:
#   ./run_server.sh                              # uses cameras.yaml, port 5000
#   ./run_server.sh --config cameras.yaml
#   ./run_server.sh --config cameras.yaml --port 8080
#
# If using GPU with onnxruntime-gpu, uncomment and adjust the line below:
#   NVIDIA_LIB="/path/to/your/site-packages/nvidia/cuXX/lib"
#   export LD_LIBRARY_PATH="${NVIDIA_LIB}:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python "${SCRIPT_DIR}/server.py" "$@"
