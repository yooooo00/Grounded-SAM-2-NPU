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
推荐两种方式，二选一：

方式 A（脚本，推荐）
```bash
export ASCEND_DEVICE_ID=0
bash scripts/run_grounded_sam2_npu.sh
```

方式 B（参数化入口）
```bash
export ASCEND_DEVICE_ID=0
python tools/run_grounded_sam2_npu.py \
  --image demo_images/truck.jpg \
  --text '$: coco' \
  --dino-cfg /abs/path/to/grounding_dino_swin-b_pretrain_obj365_goldg_v3det.py \
  --dino-weights /abs/path/to/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth \
  --sam2-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-weights checkpoints/sam2.1_hiera_large.pt \
  --device npu
```

说明：
- 输出目录：`outputs/grounded_sam2_local_demo/`
- Demo 已自动检测 NPU，并在 NPU 上使用 `autocast(bfloat16)`。
- GroundingDINO：若检测到 `third_party/groundingdino_npu/api.py` 可用，将优先走 ModelZoo NPU 实现；否则回退到原仓的纯 PyTorch 路径（可跑通但较慢）。
- 旧脚本 `scripts/run_npu_demo.sh` 已标记为过时，将转发到新的脚本。详见 `USAGE_NPU.md`。

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

## 10. 已完成工作（本仓改造）
- 设备与注意力兼容
  - SAM2 侧：去除 CUDA 专属路径；优先使用 PyTorch SDPA，若不可用则回退到显式 `QK^T`+softmax+`*V`（推理专用）。
  - Demo 侧：自动检测 NPU，启用 `autocast(bfloat16)`；仅在 CUDA 时设置 TF32。
- GroundingDINO NPU 适配挂钩
  - 新增 `adapters/groundingdino_npu_adapter.py`：在 NPU 上优先调用 ModelZoo 实现（`mmdet.apis.DetInferencer`），无可用实现时回退原仓 PyTorch 路线。
  - 新增 `third_party/groundingdino_npu/api.py`：提供最小封装（build_from_config/safe_load_state_dict/infer），并支持通过环境变量 `ENABLE_TORCHAIR_COMPILE=1` 启用 TorchAir 编译（可选）。
  - 新增 `third_party/groundingdino_npu/README.md`：说明如何放置/复用 ModelZoo 代码。
- 文档与脚本
  - 新增本文件（README_NPU.md）、`requirements-npu.txt`（仅通用依赖）、`scripts/setup_npu_env.sh`（一键安装 SAM2 侧依赖）。
  - 新增 `.gitignore` 忽略权重与 outputs。
- Demo 改造
  - `grounded_sam2_local_demo.py` 接入适配器；在 NPU 上优先走 ModelZoo 实现。

## 11. 环境迁移指引（复用你已跑通的 ModelZoo 环境）
- 不需要复制完整的 ModelZoo 仓库；关键是复用同一个 Python 环境（已安装 torch/torch_npu、torchair、mmengine/mmcv/mmdet 补丁）。
- 你可以：
  1) 直接引用你现有的 GroundingDINO 配置/权重的绝对路径（建议）。
  2) 或将以下文件复制到本仓：
     - 你的 GroundingDINO cfg（例如 `configs/mm_grounding_dino/xxx.py`，目录可自建）。
     - 对应权重 `.pth`（放在 `gdino_checkpoints/`）。
     - 可选：TorchAir 注册脚本 `register_im2col_to_torchair.py`、`register_roll_to_torchair.py`（放到 `third_party/groundingdino_npu/`），并设置 `ENABLE_TORCHAIR_COMPILE=1` 以启用编译加速。
- 其余 mmengine/mmcv/mmdet/torchair 不需要复制源码，因为已通过 pip -e 安装在你的 Python 环境中。

## 12. 下一步计划（建议）
- Demo 覆盖
  - 将仓内更多 demo（如 `grounded_sam2_dinox_demo.py` 等）统一接入适配器路径与命令行参数，方便无改代码直接运行。
- 精度/性能验证
  - 在相同图片+prompt 下，对比 NPU 与 CPU/GPU 的输出（框 IoU/掩码 IoU）；验证通过后记录一组参考指标。
  - 在 NPU 上记录开启/关闭 TorchAir 编译、channels_last、BF16 的性能数据，形成建议配置。
- 文档完善
  - 增加“命令行参数版”推理入口脚本（可直接 `--image/--text/--dino-cfg/--dino-weights/--sam2-cfg/--sam2-weights`）。
  - 根据你的实际 ModelZoo cfg/weights 路径，补充“一键命令示例”。
