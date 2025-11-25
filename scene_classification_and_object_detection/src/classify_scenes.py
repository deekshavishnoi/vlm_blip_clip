"""
Task 2 — Scene Classification (Places365, CPU-friendly)

WHAT THIS FILE DOES
-------------------
- Reads the Task 1 frame manifest (frames.csv).
- Runs a Places365 ResNet-18 scene classifier on CPU.
- Writes a tidy table (Parquet) with top-1 scene label + confidence:
    [frame_path, timestamp_sec, scene_label, confidence]

NOTES
-----
We try to load the official Places365 model via torch.hub.
If torch.hub fails (e.g., offline), we fallback to ImageNet-pretrained ResNet-18 and
map predictions to generic labels (not as good as Places365, but the pipeline still runs).
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import numpy as np

import torch.nn.functional as F


from common.io_utils import read_yaml, ensure_dir, save_parquet, init_logger

logger = init_logger("task2.scenes")


# -----------------------
# Model loading
# -----------------------

def _load_places365_resnet18() -> Tuple[torch.nn.Module, List[str], T.Compose]:
    """
    Load Places365 ResNet-18 via torch.hub along with category labels and transforms.
    """
    logger.info("Loading Places365 ResNet-18 weights + categories")
    model = torch.hub.load("CSAILVision/places365", "resnet18", pretrained=True)
    model.eval()

    # Try hub helpers first, then raw file as fallback
    classes: List[str] = []
    try:
        # Some hub versions expose this:
        classes = torch.hub.load("CSAILVision/places365", "category_places365")
    except Exception:
        try:
            classes = torch.hub.load("CSAILVision/places365", "categories")
        except Exception:
            # Final fallback: download category file (or keep generic if offline)
            import urllib.request
            url = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    txt = resp.read().decode("utf-8").strip().splitlines()
                # File format: '1  abbey' → we want 'abbey'
                classes = [line.strip().split(" ")[0][3:] for line in txt if line]
            except Exception:
                logger.warning("Failed to fetch Places365 classes; using generic labels.")
                classes = [f"class_{i}" for i in range(365)]

    transform = _make_transform(use_tencrop=False)
    return model, classes, transform

def _make_transform(use_tencrop: bool = False):
    if not use_tencrop:
        return T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    base = T.Compose([
        T.Resize(256), T.TenCrop(224),
        T.Lambda(lambda crops: torch.stack([
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(T.ToTensor()(c.convert("RGB")))
            for c in crops
        ]))
    ])
    return base

def _load_imagenet_fallback() -> Tuple[torch.nn.Module, List[str], T.Compose]:
    """
    Fallback: ImageNet ResNet-18 (not scene-specific, but keeps pipeline working).
    """
    from torchvision.models import resnet18, ResNet18_Weights

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    classes = weights.meta.get("categories", [f"class_{i}" for i in range(1000)])
    transform = weights.transforms()
    return model, classes, transform


# -----------------------
# Inference helpers
# -----------------------

@torch.inference_mode()
def _predict_topk(model: torch.nn.Module, classes: List[str], transform, image: Image.Image, k: int = 5) -> Tuple[List[str], List[float]]:
    """
    Return top-k class names and probabilities. Works with single-crop or TenCrop transforms.
    """
    x = transform(image)  # either (1,3,H,W) or (10,3,224,224)
    if x.dim() == 3:
        x = x.unsqueeze(0)

    logits = model(x)  # (N, C)
    if logits.dim() == 3:  # some torchvision models return (N, C, 1, 1)
        logits = logits.squeeze(-1).squeeze(-1)

    # If TenCrop: average over crops
    if x.size(0) > 1:
        logits = logits.mean(dim=0, keepdim=True)

    probs = F.softmax(logits, dim=1)  # (1, C)
    top_prob, top_idx = probs.topk(k, dim=1)
    top_prob = top_prob.squeeze(0).cpu().numpy().tolist()
    top_idx = top_idx.squeeze(0).cpu().numpy().tolist()
    top_labels = [classes[i] if i < len(classes) else f"class_{i}" for i in top_idx]
    return top_labels, top_prob

def _apply_confidence_floor(df: pd.DataFrame, floor: float = 0.25) -> pd.DataFrame:
    df = df.copy()
    low = df["confidence"] < floor
    df.loc[low, "scene_label"] = "uncertain"
    return df

def _temporal_smooth_by_probs(df: pd.DataFrame, labels: list[str], window: int = 5) -> pd.DataFrame:
    """Smooth using moving-average over per-label probs reconstructed from top5."""
    if "timestamp_sec" in df.columns:
        df = df.sort_values("timestamp_sec").reset_index(drop=True)

    L = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    mat = np.zeros((len(df), L), dtype=np.float32)

    # rebuild sparse probability vector from top5 lists
    for i, row in df.iterrows():
        for l, s in zip(row["top5_labels"], row["top5_probs"]):
            j = idx.get(l)
            if j is not None:
                mat[i, j] = s

    # moving average
    half = window // 2
    sm = np.copy(mat)
    for i in range(len(df)):
        lo, hi = max(0, i - half), min(len(df), i + half + 1)
        sm[i] = mat[lo:hi].mean(axis=0)

    new_idx = sm.argmax(axis=1)
    new_conf = sm.max(axis=1)
    df["scene_label_smoothed"] = [labels[j] for j in new_idx]
    df["scene_conf_smoothed"] = new_conf.astype(float)
    return df


def classify_scenes(
    frames_csv: str | Path,
    out_parquet: str | Path,
    use_places365: bool = True,
    max_images: int | None = None,
) -> Path:
    """
    Classify scenes for frames listed in frames_csv, save to Parquet.

    Columns written:
        frame_path, timestamp_sec, scene_label, confidence
    """
    frames_csv = Path(frames_csv)
    out_parquet = Path(out_parquet)
    ensure_dir(out_parquet.parent)

    df = pd.read_csv(frames_csv)
    if max_images is not None:
        df = df.head(max_images)

    if df.empty:
        logger.warning("No frames found in %s", frames_csv)
        out_df = pd.DataFrame(columns=["frame_path", "timestamp_sec", "scene_label", "confidence"])
        save_parquet(out_df, out_parquet)
        return out_parquet

    # Load model
    classes: List[str]
    try:
        if use_places365:
            model, classes, transform = _load_places365_resnet18()
            logger.info("Loaded Places365 ResNet-18 via torch.hub")
        else:
            raise RuntimeError("Forcing fallback")
    except Exception as e:
        logger.warning("Falling back to ImageNet ResNet-18 due to: %s", e)
        model, classes, transform = _load_imagenet_fallback()
        logger.info("Loaded ImageNet ResNet-18 fallback")

    rows = []
    for _, row in df.iterrows():
        p = Path(row["frame_path"])
        ts = float(row["timestamp_sec"])
        try:
            with Image.open(p) as im:
                top_labels, top_probs = _predict_topk(model, classes, transform, im, k=5)
                label, conf = top_labels[0], float(top_probs[0])
        except Exception as e:
            logger.warning("Failed on image %s: %s", p, e)
            label, conf = "unknown", 0.0

        rows.append({
            "frame_path": str(p),
            "timestamp_sec": ts,
            "scene_label": label,
            "confidence": conf,
            "top5_labels": top_labels,
            "top5_probs": [float(x) for x in top_probs],
        })

    out_df = pd.DataFrame(rows)

    # after building out_df with: scene_label, confidence, top5_labels, top5_probs
    out_df = _temporal_smooth_by_probs(out_df, classes, window=5)

    # LOWER the floor (night scenes are naturally low-conf)
    FLOOR = 0.15
    # Use smoothed label unless even smoothed confidence is very low
    use_smoothed = out_df["scene_conf_smoothed"] >= FLOOR
    out_df["scene_label_final"] = np.where(use_smoothed, out_df["scene_label_smoothed"], out_df["scene_label"])

    # mark truly low-confidence frames as 'uncertain'
    out_df.loc[out_df["scene_conf_smoothed"] < FLOOR, "scene_label_final"] = "uncertain"

    save_parquet(out_df, out_parquet)
    logger.info("Wrote scene predictions (smoothed) → %s (rows=%d)", out_parquet, len(out_df))

    return out_parquet


# -----------------------
# Tiny CLI
# -----------------------

def main():

    import argparse
    parser = argparse.ArgumentParser(description="Task 2: Scene classification over frames (Places365/ResNet-18).")
    parser.add_argument("--cfg", required=True, help="Path to task2.yaml")
    parser.add_argument("--max_images", type=int, default=None, help="Limit frames for a quick run.")
    args = parser.parse_args()

    cfg = read_yaml(args.cfg)
    classify_scenes(
        frames_csv=cfg["frames_csv"],
        out_parquet=cfg["scenes_path"],
        use_places365=True,  # try Places365 first
        max_images=args.max_images or cfg.get("max_images", None),
    )


if __name__ == "__main__":
    main()
