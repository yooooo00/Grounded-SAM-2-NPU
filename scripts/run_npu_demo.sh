#!/usr/bin/env bash
set -euo pipefail

# 简易 NPU 启动脚本：运行本仓库自带的 grounded_sam2_local_demo.py
# 需提前准备：
# - Ascend/CANN + torch + torch_npu（版本按官方适配 Readme 交集）
# - SAM2 与 GroundingDINO 的权重与配置，放至默认路径或修改脚本中的常量

export ASCEND_DEVICE_ID=${ASCEND_DEVICE_ID:-0}
echo "ASCEND_DEVICE_ID=${ASCEND_DEVICE_ID}"

# 可选：设置 PyTorch NPU AMP 精度，默认使用 BF16（由代码内部 autocast 控制）

python grounded_sam2_local_demo.py

echo "Done. Check outputs/grounded_sam2_local_demo for results."

