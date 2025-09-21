This folder is reserved for integrating the GroundingDINO Ascend (NPU) implementation from ModelZoo-PyTorch.

Expected files and API (minimal):

- api.py providing:
  - build_from_config(cfg_path) -> model
  - safe_load_state_dict(model, state_dict) -> (missing_keys, unexpected_keys)
  - infer(model, image_tensor, caption, box_threshold, text_threshold) -> dict
    Required keys in dict: 'boxes_cxcywh' (Tensor [N,4]); optional: 'scores' (Tensor [N]), 'labels' (List[str])

Place only the minimal files required for inference to keep the repository lean. Follow the original LICENSE of the integrated code.

