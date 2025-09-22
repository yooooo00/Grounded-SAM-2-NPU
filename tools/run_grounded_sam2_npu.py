import os
import argparse
import torch

from torchvision.ops import box_convert

from grounding_dino.groundingdino.util.inference import load_image
from adapters.groundingdino_npu_adapter import build_model as build_dino_model, predict as dino_predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def parse_args():
    p = argparse.ArgumentParser("Grounded-SAM-2 NPU runner")
    p.add_argument("--image", required=True, type=str, help="input image path")
    p.add_argument("--text", required=True, type=str, help="text prompt, e.g. 'car. tire.' or '$: coco'")
    p.add_argument("--dino-cfg", required=True, type=str, help="GroundingDINO config path (ModelZoo cfg supported)")
    p.add_argument("--dino-weights", required=True, type=str, help="GroundingDINO checkpoint (.pth)")
    p.add_argument("--sam2-cfg", required=True, type=str, help="SAM2 config yaml path")
    p.add_argument("--sam2-weights", required=True, type=str, help="SAM2 weights path (*.pt)")
    p.add_argument("--device", default="npu", choices=["npu", "cpu", "cuda"], help="device (default npu)")
    p.add_argument("--box-threshold", type=float, default=0.35, help="box score threshold")
    p.add_argument("--text-threshold", type=float, default=0.25, help="text score threshold")
    p.add_argument("--enable-torchair-compile", action="store_true", help="enable TorchAir compile for GroundingDINO")
    return p.parse_args()


def main():
    args = parse_args()

    if args.enable_torchair_compile:
        os.environ["ENABLE_TORCHAIR_COMPILE"] = "1"

    device = torch.device(args.device)

    # Build SAM2 predictor
    sam2_model = build_sam2(args.sam2_cfg, ckpt_path=args.sam2_weights, device=args.device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # Build GroundingDINO (NPU: prefer ModelZoo adapter)
    dino_model = build_dino_model(args.dino_cfg, args.dino_weights, device)

    # Load image and run GroundingDINO detection
    image_source, image = load_image(args.image)
    boxes, scores, labels = dino_predict(
        model=dino_model,
        image=image,
        caption=args.text,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=device,
    )

    # Prepare boxes for SAM2
    h, w, _ = image_source.shape
    boxes_abs = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype)
    input_boxes = box_convert(boxes=boxes_abs, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

    # Run SAM2 predictor
    sam2_predictor.set_image(image_source)
    masks, ious, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes if input_boxes.shape[0] > 0 else None,
        multimask_output=False,
    )

    print(f"Detections: {len(input_boxes)} boxes; SAM2 masks shape: {masks.shape if masks is not None else None}")


if __name__ == "__main__":
    main()

