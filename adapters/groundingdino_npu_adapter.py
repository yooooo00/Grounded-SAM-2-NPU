import os
from typing import Tuple, List

import torch


def _has_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _has_modelzoo_impl() -> bool:
    try:
        # expected user-provided integration under third_party
        import importlib
        importlib.import_module("third_party.groundingdino_npu")
        return True
    except Exception:
        return False


def build_model(cfg_path: str, weights_path: str, device: torch.device):
    """Build GroundingDINO for NPU path.

    Priority:
    1) If third_party.groundingdino_npu is present, use it.
    2) Fallback to repository builtin loader (works with PyTorch fallback on NPU).
    """
    if _has_npu() and _has_modelzoo_impl():
        from third_party.groundingdino_npu import api as gdino_api  # type: ignore
        model = gdino_api.build_from_config(cfg_path)
        state = torch.load(weights_path, map_location="cpu")
        # try common keys; users may need to adapt to their integrated version
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = gdino_api.safe_load_state_dict(model, state)  # user to implement
        if missing or unexpected:
            print("[GroundingDINO NPU adapter] State dict mismatches:",
                  {"missing": missing, "unexpected": unexpected})
        model.eval().to(device)
        return model

    # Fallback to original inference loader
    from grounding_dino.groundingdino.util.inference import load_model as _load
    return _load(cfg_path, weights_path, device=str(device))


def predict(
    model,
    image: torch.Tensor,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Run detection with GroundingDINO on NPU.

    If ModelZoo integration exists, call its API; otherwise fallback to
    builtin predict (which uses PyTorch fallback ops on NPU).
    Returns (boxes[cxcywh], scores, labels)
    """
    # Normalize caption format: lower + trailing dot
    cap = caption.strip().lower()
    if not cap.endswith("."):
        cap += "."

    if _has_npu() and _has_modelzoo_impl():
        from third_party.groundingdino_npu import api as gdino_api  # type: ignore
        with torch.no_grad():
            out = gdino_api.infer(
                model=model,
                image=image.to(device),
                caption=cap,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        # expected keys from user-integrated API
        boxes = out["boxes_cxcywh"]
        scores = out.get("scores", torch.ones(boxes.size(0), device=boxes.device))
        labels = out.get("labels", ["object"] * boxes.size(0))
        return boxes, scores, labels

    # Fallback – use builtin inference path
    from grounding_dino.groundingdino.util.inference import predict as _predict
    boxes, logits, phrases = _predict(
        model=model,
        image=image,
        caption=cap,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=str(device),
    )
    scores = logits  # logits already filtered by threshold inside builtin path
    return boxes, scores, phrases

