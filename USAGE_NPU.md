# Grounded‑SAM‑2 NPU 使用方法（新版入口）

本说明配合 `README_NPU.md`，给出两种推荐启动方式：

- 脚本方式：`scripts/run_grounded_sam2_npu.sh`
- 参数化入口：`tools/run_grounded_sam2_npu.py`

在此之前，请确保你已按 `README_NPU.md` 完成依赖安装与权重准备，并且复用了你已跑通的 ModelZoo GroundingDINO 环境（含 torch/torch_npu、torchair、mmengine/mmcv/mmdet 补丁）。

## 一、脚本方式（推荐）

1) 设置（如需）TorchAir 编译以加速 GroundingDINO（可选）
- 在运行前设置：`export ENABLE_TORCHAIR_COMPILE=1`
- 如需注册脚本，将 ModelZoo 的 `register_im2col_to_torchair.py`、`register_roll_to_torchair.py` 复制到 `third_party/groundingdino_npu/`

2) 执行脚本（修改权重与配置路径）
```bash
export ASCEND_DEVICE_ID=0
# 按需修改以下变量（脚本内也可通过环境变量覆盖）
bash scripts/run_grounded_sam2_npu.sh
```
脚本默认变量：
- `IMAGE`：输入图片，默认 `demo_images/truck.jpg`
- `TEXT`：文本 prompt，默认 `"car. tire."` 或可用 `'$: coco'`
- `DINO_CFG`：ModelZoo 的 mmengine/mmdet 配置（示例已给出）
- `DINO_WEIGHTS`：对应权重 `.pth`（请改为你的实际绝对路径）
- `SAM2_CFG`：`configs/sam2.1/sam2.1_hiera_l.yaml`
- `SAM2_WEIGHTS`：`checkpoints/sam2.1_hiera_large.pt`

## 二、参数化入口（直接命令行）

示例：
```bash
export ASCEND_DEVICE_ID=0
python tools/run_grounded_sam2_npu.py \
  --image demo_images/truck.jpg \
  --text '$: coco' \
  --dino-cfg /abs/path/to/grounding_dino_swin-b_pretrain_obj365_goldg_v3det.py \
  --dino-weights /abs/path/to/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth \
  --sam2-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-weights checkpoints/sam2.1_hiera_large.pt \
  --device npu \
  --enable-torchair-compile  # 可选
```
参数说明：
- `--image/--text`：输入图片与文本 prompt（GroundingDINO 要求小写+句号结尾或 `'$: coco'`）。
- `--dino-cfg/--dino-weights`：指定 ModelZoo 的配置与权重。
- `--sam2-cfg/--sam2-weights`：指定 SAM2 配置与权重。
- `--device`：默认 `npu`。
- `--enable-torchair-compile`：启用 TorchAir 编译加速（需环境已安装 torchair）。

## 三、输出
- 运行完成后在 `outputs/`（或终端）可查看检测框数量与 SAM2 掩码输出情况。
- 如需可视化叠加/JSON 导出，可复用 `grounded_sam2_local_demo.py` 或按需在 `tools/run_grounded_sam2_npu.py` 中添加保存逻辑。

## 四、常见问题
- `ModuleNotFoundError: npu_utils`：旧 demo 脚本残留。请使用 `tools/run_grounded_sam2_npu.py` 或最新的 `grounded_sam2_local_demo.py`（已去除该依赖）。
- `Failed to load custom C++ ops`：可忽略（CUDA 扩展）。在 NPU 路径下我们通过 ModelZoo 实现进行推理。
- 打包/安装问题：请严格参考 `README_NPU.md` 的版本组合与安装脚本。
