#!/usr/bin/env bash
set -euo pipefail

# Example launcher for Grounded‑SAM‑2 on Ascend NPU.
# Adjust the paths below to your environment, or pass via env/CI.

: "${ASCEND_DEVICE_ID:=0}"
export ASCEND_DEVICE_ID

# Optionally enable TorchAir compile for GroundingDINO
# export ENABLE_TORCHAIR_COMPILE=1

IMAGE=${IMAGE:-"demo_images/truck.jpg"}
TEXT=${TEXT:-"car. tire."}

# If you want to reuse ModelZoo cfg/weights, point these to your absolute paths
DINO_CFG=${DINO_CFG:-"configs/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det.py"}
DINO_WEIGHTS=${DINO_WEIGHTS:-"/root/work/filestorage/cyy/ModelZoo-PyTorch/ACL_PyTorch/built-in/cv/GroundingDINO/mmdetection/weights/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth"}

# SAM2 config/weights
SAM2_CFG=${SAM2_CFG:-"configs/sam2.1/sam2.1_hiera_l.yaml"}
SAM2_WEIGHTS=${SAM2_WEIGHTS:-"checkpoints/sam2.1_hiera_large.pt"}

python tools/run_grounded_sam2_npu.py \
  --image "${IMAGE}" \
  --text "${TEXT}" \
  --dino-cfg "${DINO_CFG}" \
  --dino-weights "${DINO_WEIGHTS}" \
  --sam2-cfg "${SAM2_CFG}" \
  --sam2-weights "${SAM2_WEIGHTS}" \
  --device npu

