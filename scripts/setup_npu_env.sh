#!/usr/bin/env bash
set -euo pipefail

# 本脚本仅补齐 Grounded‑SAM‑2 对 SAM2 所需的通用依赖
# 假定你已在当前环境按 ModelZoo GroundingDINO 的 README
# 安装好了 torch/torch_npu、torchair、mmengine/mmcv/mmdet（含补丁）。

python -m pip install -U pip
pip install -r requirements-npu.txt

# 可选：将本仓以可编辑方式安装，避免 import 问题
pip install -e .

echo "NPU env (SAM2 deps) installed."

