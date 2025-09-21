"""GroundingDINO NPU API (ModelZoo integration wrapper).

This file adapts the ModelZoo-PyTorch GroundingDINO (TorchAir) implementation
to a minimal API used by this repo's adapter:

- build_from_config(cfg_path) -> model_like
- safe_load_state_dict(model_like, state_dict) -> (missing, unexpected)
- infer(model_like, image, caption, box_threshold, text_threshold) -> dict

Requirements (as per ModelZoo README/install script):
- torch==2.1.0, torch_npu==2.1.0.post10, torchair
- mmengine==0.10.6 (patched), mmcv==2.1.0 (patched), mmdetection@cfd5d3a9 (patched)

Note: This is a thin wrapper. It relies on mmdet.apis.DetInferencer.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch


def _ensure_packages():
    try:
        import mmdet  # noqa: F401
        import mmengine  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "ModelZoo GroundingDINO NPU requires patched mmengine/mmcv/mmdet. "
            "Please follow ModelZoo install_requirements.sh to install."
        ) from e


class _InferencerHolder:
    def __init__(self, cfg_path: str, weights_path: str | None, device: str = "npu") -> None:
        _ensure_packages()
        from mmdet.apis import DetInferencer

        # DetInferencer accepts either config path or None with weights
        init_kwargs = {"device": device}
        if cfg_path and cfg_path.strip():
            init_kwargs["model"] = cfg_path
        if weights_path and weights_path.strip():
            init_kwargs["weights"] = weights_path

        self.inferencer = DetInferencer(**init_kwargs)
        # Optional: TorchAir compile path (for performance). Enable by env.
        if os.environ.get("ENABLE_TORCHAIR_COMPILE", "0") == "1":
            try:
                import torchair as tng
                from torchair.configs.compiler_config import CompilerConfig
                # optional registration (if user copied these helpers in place)
                try:
                    import third_party.groundingdino_npu.register_im2col_to_torchair  # noqa: F401
                    import third_party.groundingdino_npu.register_roll_to_torchair  # noqa: F401
                except Exception:
                    pass

                config = CompilerConfig()
                config.experimental_config.frozen_parameter = True
                config.experimental_config.tiling_schedule_optimize = True
                npu_backend = tng.get_npu_backend(compiler_config=config)

                # compile common submodules if present
                m = self.inferencer.model
                for name in ["backbone", "encoder", "decoder", "language_model"]:
                    if hasattr(m, name):
                        sub = getattr(m, name)
                        setattr(m, name, torch.compile(sub, dynamic=False, fullgraph=True, backend=npu_backend))
                        try:
                            tng.use_internal_format_weight(getattr(m, name))
                        except Exception:
                            pass
            except Exception as e:
                print("[GroundingDINO NPU api] TorchAir compile disabled due to:", e)


def build_from_config(cfg_path: str, weights_path: str | None = None):
    """Construct DetInferencer holder with config and optional weights.

    Passing weights here aligns with ModelZoo usage and avoids manual state_dict load.
    """
    return _InferencerHolder(cfg_path, weights_path=weights_path, device="npu")


def safe_load_state_dict(model_like: _InferencerHolder, state_dict) -> Tuple[List[str], List[str]]:
    """For DetInferencer, weights are typically passed at construction.
    We provide a fallback attempt to load from state_dict if needed.
    Returns (missing_keys, unexpected_keys).
    """
    # If the caller passed a full checkpoint dict, try common keys
    if isinstance(state_dict, dict):
        inner = state_dict.get("model", state_dict)
    else:
        inner = state_dict

    # If the user prefers path-based load, they should build with weights
    try:
        model = model_like.inferencer.model
        missing, unexpected = model.load_state_dict(inner, strict=False)
        return list(missing), list(unexpected)
    except Exception:
        # Fall back to no-op; ModelZoo README suggests using weights at init
        return [], []


def _to_hwc_uint8(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        x = image.detach().cpu()
        # expect 3xHxW in [0,1] or [0,255]
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.squeeze(0) if x.shape[0] == 1 else x
            x = x.permute(1, 2, 0)
        x = x.numpy()
    else:
        x = image

    if x.dtype != np.uint8:
        x = np.clip(x, 0, 1) * 255.0 if x.max() <= 1.0 else np.clip(x, 0, 255)
        x = x.astype(np.uint8)
    return x


def infer(
    model_like: _InferencerHolder,
    image: torch.Tensor | np.ndarray,
    caption: str,
    box_threshold: float,
    text_threshold: float,
) -> Dict:
    """Run GroundingDINO inference via mmdet DetInferencer.

    Returns a dict with keys:
      - boxes_cxcywh: Tensor [N,4] normalized to [0,1] in cxcywh
      - scores: Tensor [N]
      - labels: List[str]
    """
    img = _to_hwc_uint8(image)

    # Normalize caption format used by Grounded-SAM-2: lower + trailing dot
    cap = caption.strip().lower()
    if not cap.endswith("."):
        cap += "."

    # DetInferencer expects prompts via kwargs; texts can be list/tuple
    call_kwargs = dict(
        inputs=img,
        texts=cap,
        pred_score_thr=box_threshold,
        return_datasamples=True,
        out_dir="",  # avoid saving
        show=False,
        print_result=False,
    )

    with torch.no_grad():
        result = model_like.inferencer(**call_kwargs)

    # Parse results
    # mmdet outputs are in result["predictions"]; each with bboxes (xyxy) and scores, labels
    preds = result.get("predictions", [])
    if len(preds) == 0:
        # Fallback key
        preds = result.get("predictions", result)

    # Simple parser for single-image case
    if isinstance(preds, list) and len(preds) >= 1:
        p = preds[0]
        bboxes_xyxy = np.asarray(p.get("bboxes", []), dtype=np.float32)
        scores = np.asarray(p.get("scores", []), dtype=np.float32)
        labels = p.get("labels", [])
    else:
        bboxes_xyxy = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        labels = []

    h, w = img.shape[:2]
    # Convert xyxy (abs) -> cxcywh normalized
    if bboxes_xyxy.size > 0:
        x1, y1, x2, y2 = [bboxes_xyxy[:, i] for i in range(4)]
        cx = (x1 + x2) * 0.5 / float(w)
        cy = (y1 + y2) * 0.5 / float(h)
        ww = (x2 - x1) / float(w)
        hh = (y2 - y1) / float(h)
        boxes_cxcywh = torch.from_numpy(
            np.stack([cx, cy, ww, hh], axis=-1)
        )
    else:
        boxes_cxcywh = torch.zeros((0, 4), dtype=torch.float32)

    return {
        "boxes_cxcywh": boxes_cxcywh,
        "scores": torch.from_numpy(scores),
        "labels": labels,
    }
