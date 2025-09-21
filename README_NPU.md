# Ascend NPU 安装与运行指南（推理）

本文是 Grounded‑SAM‑2 在华为昇腾（Ascend）NPU 上的可操作安装与运行说明。默认你已在 NPU 服务器上按 Gitee ModelZoo 的 GroundingDINO 文档成功跑通（TorchAir + mmengine/mmcv/mmdet 打补丁）。

## 目标
- 只做推理（不训练）。
- 直接复用你已有的 ModelZoo GroundingDINO 环境；补齐 SAM2 依赖与权重即可。
- 在 NPU 上优先走 GroundingDINO 的 ModelZoo 实现；SAM2 采用“去 CUDA 化 + SDPA/显式注意力回退”的路径。

## 1. 环境与版本
- 硬件/OS：Ascend NPU（910/910B/Atlas 等），Linux 发行版（Ubuntu/CentOS 等）。
- CANN、torch、torch_npu：与 ModelZoo GroundingDINO README 一致的组合（示例：torch==2.1.0、torch_npu==2.1.0.post10）。
- 已安装组件：torchair、mmengine(0.10.6+补丁)、mmcv(2.1.0+补丁)、mmdetection@cfd5d3a9(+补丁)。
- 不安装：xformers、flash-attn、triton（CUDA 专属）。

## 2. 克隆并进入仓库
```bash
# SSH 示例
git clone git@github.com:yooooo00/Grounded-SAM-2-NPU.git
cd Grounded-SAM-2-NPU
```

## 3. 安装 SAM2 侧依赖（保留你已装好的 ModelZoo 依赖）
```bash
# 可选：升级 pip
python -m pip install -U pip

# 安装仅与 SAM2/本仓运行相关的通用依赖（不包含 torch/torch_npu）
# 也可使用 requirements 文件：requirements-npu.txt（见仓库根）
pip install -r requirements-npu.txt

# 或等价安装：
# pip install hydra-core omegaconf einops Pillow tqdm opencv-python supervision \
#             pycocotools yacs huggingface-hub

# 可选：安装本仓为可编辑包，避免 import 问题
pip install -e .
```

## 4. 权重与配置
- SAM2：
  - 配置：`configs/sam2.1/sam2.1_hiera_l.yaml`
  - 权重：`checkpoints/sam2.1_hiera_large.pt`
- GroundingDINO：二选一
  - A. 使用本仓默认 Demo 路径：`grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py` + `gdino_checkpoints/groundingdino_swint_ogc.pth`
  - B. 使用 ModelZoo 的 mmengine/mmdet 配置与权重（推荐）：如 `configs/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det.py` + 对应 `.pth`。
    - 本仓已内置适配器，会通过 `mmdet.apis.DetInferencer` 路径优先调用 ModelZoo 实现。

## 5. 运行（先跑通）
```bash
export ASCEND_DEVICE_ID=0
bash scripts/run_npu_demo.sh
```
- 输出目录：`outputs/grounded_sam2_local_demo/`
- Demo 已自动检测 NPU，并在 NPU 上使用 `autocast(bfloat16)`。
- GroundingDINO：若检测到 `third_party/groundingdino_npu/api.py` 可用，将优先走 ModelZoo NPU 实现；否则回退到原仓的纯 PyTorch 路径（可跑通但较慢）。

## 6. （可选）集成 ModelZoo GroundingDINO NPU 实现
- 放置路径：`third_party/groundingdino_npu/`
- 预期 API（已在本仓提供封装骨架）：`third_party/groundingdino_npu/api.py` 导出：
  - `build_from_config(cfg_path) -> model_like`
  - `safe_load_state_dict(model_like, state_dict) -> (missing_keys, unexpected_keys)`
  - `infer(model_like, image_tensor, caption, box_threshold, text_threshold) -> { 'boxes_cxcywh': Tensor, 'scores': Tensor, 'labels': List[str] }`
- 说明：当前 `api.py` 已基于 `mmdet.apis.DetInferencer` 提供最小封装；在你的 ModelZoo 环境下可直接使用。

## 7. 常见问题与排查
- ImportError: mmdet/mmcv/mmengine/torchair 未找到：请复用你已跑通 ModelZoo 的环境，或按其 install_requirements.sh（clone + checkout + 打补丁 + pip 安装）安装。
- SDPA/FlashAttention 报错：本仓已关闭 CUDA 专属内核；优先用 PyTorch SDPA，若不支持则回退到显式 `QK^T` + softmax + `*V`（推理专用，性能一般但可用）。
- 权重/路径不存在：编辑 `grounded_sam2_local_demo.py` 顶部默认路径，或将文件放入相应目录。
- 不要将权重加入 git：`checkpoints/`、`gdino_checkpoints/` 应加入 .gitignore，由服务器侧单独下发。

## 8. 快速安装脚本（可选）
- 使用 `scripts/setup_npu_env.sh` 一键安装第 3 步依赖：
```bash
bash scripts/setup_npu_env.sh
```
- 该脚本不会安装 torch/torch_npu、torchair、mmengine/mmcv/mmdet。它们请继续复用你的 ModelZoo 环境。

## 9. 目录提示
- GroundingDINO 的 NPU 实现放到：`third_party/groundingdino_npu/`（已含 `api.py`）。
- 适配器：`adapters/groundingdino_npu_adapter.py`（自动优先选择 NPU 实现）。
- Demo：`grounded_sam2_local_demo.py`（端到端：文本检测 + SAM2 分割）。

