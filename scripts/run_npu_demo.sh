#!/usr/bin/env bash
set -euo pipefail

# Deprecated: please use scripts/run_grounded_sam2_npu.sh or
# tools/run_grounded_sam2_npu.py instead. This script will forward
# to the new launcher for backward compatibility.

export ASCEND_DEVICE_ID=${ASCEND_DEVICE_ID:-0}
echo "[DEPRECATED] scripts/run_npu_demo.sh -> forwarding to scripts/run_grounded_sam2_npu.sh"

DIR=$(cd "$(dirname "$0")" && pwd)
"${DIR}/run_grounded_sam2_npu.sh"
