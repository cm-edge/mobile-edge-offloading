# shared/models.py
# torchvision-only ImageNet classification helper (MobileNetV2/V3, EfficientNet B0/B2/B4/B5/B6)
import os
import time
from io import BytesIO
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.models as M


# ------------------------------
# Optional: ImageNet label names
# ------------------------------
_IMAGENET_LABELS: Optional[List[str]] = None

def _load_imagenet_labels() -> List[str]:
    """
    Load human-readable ImageNet-1k class names if shared/imagenet_class_index.json exists.
    Otherwise, fall back to ['class_0', ..., 'class_999'].
    """
    global _IMAGENET_LABELS
    if _IMAGENET_LABELS is not None:
        return _IMAGENET_LABELS
    labels_path = os.path.join(os.path.dirname(__file__), "imagenet_class_index.json")
    if os.path.exists(labels_path):
        import json
        with open(labels_path, "r", encoding="utf-8") as f:
            j = json.load(f)  # {"0": ["n01440764","tench"], ...}
        _IMAGENET_LABELS = [j[str(i)][1] for i in range(1000)]
    else:
        _IMAGENET_LABELS = [f"class_{i}" for i in range(1000)]
    return _IMAGENET_LABELS


# ------------------------------
# Supported models (torchvision)
# ------------------------------
# Friendly name -> (ctor, weights_enum)
# You can add more aliases if you like (e.g., "mv3s", "effb4", etc.)
def _resolve_torchvision(name: str):
    name = name.lower()
    if name in ("mobilenet_v3_small", "mv3_small", "mv3s"):
        w = M.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        return M.mobilenet_v3_small, w
    #if name in ("mobilenet_v3_large", "mv3_large", "mv3l"):
        #w = M.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        #return M.mobilenet_v3_large, w
    #if name in ("mobilenet_v2", "mv2"):
        #w = M.MobileNet_V2_Weights.IMAGENET1K_V1
        #return M.mobilenet_v2, w

    if name in ("efficientnet_b0", "eff0", "b0"):
        w = M.EfficientNet_B0_Weights.IMAGENET1K_V1
        return M.efficientnet_b0, w
    if name in ("efficientnet_b1", "eff1", "b1"):
        w =  M.EfficientNet_B1_Weights.IMAGENET1K_V2
        return M.efficientnet_b1, w

    if name in ("efficientnet_b2", "eff2", "b2"):
        w = M.EfficientNet_B2_Weights.IMAGENET1K_V1
        return M.efficientnet_b2, w
    
    if name in ("efficientnet_b3", "eff3", "b3"):
        w = M.EfficientNet_B3_Weights.IMAGENET1K_V1
        return M.efficientnet_b3, w
    if name in ("efficientnet_b4", "eff4", "b4"):
        w = M.EfficientNet_B4_Weights.IMAGENET1K_V1
        return M.efficientnet_b4, w
    if name in ("efficientnet_b5", "eff5", "b5"):
        w = M.EfficientNet_B5_Weights.IMAGENET1K_V1
        return M.efficientnet_b5, w
    if name in ("efficientnet_b6", "eff6", "b6"):
        w = M.EfficientNet_B6_Weights.IMAGENET1K_V1
        return M.efficientnet_b6, w

    raise ValueError(
        "Unsupported model. Try one of: "
        "mobilenet_v3_small"
        "efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6"
    )


def build_model(
    model_name: str,
    precision: str = "fp32",   # "fp32" | "fp16" (fp16 only meaningful on CUDA)
    device: Optional[str] = None,  # "cuda" | "cpu"
):
    """
    Load a torchvision ImageNet classifier with official pretrained weights.

    Args:
      model_name: str, e.g. "mobilenet_v3_small", "efficientnet_b4"
      precision: "fp32" or "fp16" (fp16 only effective on CUDA)
      device: "cuda" or "cpu" (default: auto-detect CUDA if available)

    Returns:
      model: torch.nn.Module in eval() mode, moved to device
      preprocess: torchvision transform for resize/crop/normalize
      device_str: "cuda" or "cpu"
    """
    ctor, weights = _resolve_torchvision(model_name)

    # Auto device selection
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate model with official pretrained weights
    model = ctor(weights=weights).eval().to(device)
    preprocess = weights.transforms()

    # Optional half precision on CUDA
    if precision.lower() == "fp16" and device == "cuda":
        model = model.half()

    return model, preprocess, device



@torch.no_grad()
def infer_image_bytes(
    model,
    preprocess,
    device: str,
    img_bytes: bytes,
    precision: str = "fp32",
    topk: int = 5,
    warmup: int = 0,   # set >0 to stabilize timing on first runs
) -> Dict:
    """
    Run single-image inference from raw bytes and return top-k with timings (ms).
    """
    # Decode
    img = Image.open(BytesIO(img_bytes)).convert("RGB")

    # Optional warmup (useful on GPU)
    if warmup > 0:
        dummy = preprocess(img).unsqueeze(0).to(device)
        if precision.lower() == "fp16" and device == "cuda":
            dummy = dummy.half()
        for _ in range(warmup):
            _ = model(dummy)

    # Preprocess timing
    t0 = time.perf_counter()
    x = preprocess(img)  # [C,H,W]
    if precision.lower() == "fp16" and device == "cuda":
        x = x.half()
    x = x.unsqueeze(0).to(device)  # [1,C,H,W]
    t1 = time.perf_counter()

    # Inference timing
    y = model(x)
    t2 = time.perf_counter()

    # Softmax & top-k on CPU for convenience
    probs = F.softmax(y, dim=1).squeeze(0).float().cpu()
    k = min(topk, probs.numel())
    p, idx = torch.topk(probs, k)

    labels = _load_imagenet_labels()
    top_list = []
    for conf, i in zip(p.tolist(), idx.tolist()):
        name = labels[i] if i < len(labels) else f"class_{i}"
        top_list.append({"index": int(i), "label": name, "prob": float(conf)})

    return {
        "top1": top_list[0] if top_list else None,
        "topk": top_list,
        "timing_ms": {
            "preprocess": (t1 - t0) * 1000.0,
            "infer": (t2 - t1) * 1000.0,
            "total": (t2 - t0) * 1000.0,
        },
    }
