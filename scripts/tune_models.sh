#!/bin/bash
# Compatible entry point. Python stdlib captures logs/banks, verifies the
# actual device, and publishes a candidate without changing deployed models.
# Usage: bash scripts/tune_models.sh --device 2 --output-name vision-encoder-trt-tuned
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "${SCRIPT_DIR}/tune_vision.py" "$@"
