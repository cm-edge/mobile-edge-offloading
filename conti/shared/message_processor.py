# shared/message_processor.py
# NOTE: This version intentionally reloads the model on EVERY request (no caching).
# Useful when you want to measure cold-start energy/latency repeatedly.

import os
import base64
from typing import Dict, Any

from .models import build_model, infer_image_bytes


def _get_cfg_from_env() -> Dict[str, Any]:
    """
    Read runtime configuration from environment variables.
    """
    return {
        "model_name": os.getenv("MODEL_NAME", "mobilenet_v3_small").strip().lower(),
        "precision": os.getenv("MODEL_PRECISION", "fp32").strip().lower(),  # "fp32" | "fp16"
        "device": os.getenv("MODEL_DEVICE", None),  # "cuda" | "cpu" | None (auto)
        "topk": int(os.getenv("MODEL_TOPK", "5")),
        "warmup": int(os.getenv("INFER_WARMUP", "0")),  # warmup iterations; keep 0 for cold-start
    }


def process_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single inference request.

    Expected payload:
    {
      "client_id": "...",
      "request_id": "...",
      "payload_type": "image",
      "image_b64": "<BASE64 STRING>"
    }

    Returns:
    {
      "ok": true,
      "model": "<model_name>",
      "precision": "fp32|fp16",
      "top1": {...},
      "top5": [...],
      "timing_ms": {
        "preprocess": ...,
        "infer": ...,
        "total": ...
      },
      "request_id": "...",     # echoed if present
      "client_id": "..."       # echoed if present
    }
    """
    try:
        # 1) Validate payload and decode image
        if payload.get("payload_type") != "image":
            return {"ok": False, "error": f"unsupported payload_type: {payload.get('payload_type')}"}

        img_b64 = payload.get("image_b64")
        if not img_b64:
            return {"ok": False, "error": "missing image_b64"}

        try:
            img_bytes = base64.b64decode(img_b64, validate=True)
        except Exception as e:
            return {"ok": False, "error": f"invalid image_b64: {e}"}

        # 2) Read config (no caching): load model + weights on EVERY request
        cfg = _get_cfg_from_env()
        model, preprocess, device = build_model(
            model_name=cfg["model_name"],
            precision=cfg["precision"],
            device=cfg["device"],
        )

        # 3) Inference
        result = infer_image_bytes(
            model=model,
            preprocess=preprocess,
            device=device,
            img_bytes=img_bytes,
            precision=cfg["precision"],
            topk=max(1, int(cfg["topk"])),
            warmup=max(0, int(cfg["warmup"])),  # keep 0 for strict cold-start
        )

        # 4) Compose response
        out = {
            "ok": True,
            "model": cfg["model_name"],
            "precision": cfg["precision"],
            "top1": result.get("top1"),
            "top5": result.get("topk"),
            "timing_ms": result.get("timing_ms"),
        }
        if "request_id" in payload:
            out["request_id"] = payload["request_id"]
        if "client_id" in payload:
            out["client_id"] = payload["client_id"]

        return out

    except Exception as e:
        return {"ok": False, "error": f"inference_failed: {e}"}
