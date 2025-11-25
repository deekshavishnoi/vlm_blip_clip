"""
Task 2 — Composite Visualizations (Frame + Boxes + Caption + Scene + Objects)

Creates single images per sample that combine:
  1) The frame with YOLO detections (boxes + labels),
  2) A caption strip below with: Caption (Task 1), Scene (Task 2), Objects list, Timestamp/filename.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common.io_utils import ensure_dir, init_logger, read_yaml, load_jsonl

logger = init_logger("task2.viz")

FONT = cv2.FONT_HERSHEY_SIMPLEX

def _denorm_xyxy(row: pd.Series, w: int, h: int):
    """
    Our detection parquet stores normalized coords in [0,1]. Convert to pixels.
    """
    x1 = int(max(0, min(1, float(row["x1"]))) * w)
    x2 = int(max(0, min(1, float(row["x2"]))) * w)
    y1 = int(max(0, min(1, float(row["y1"]))) * h)
    y2 = int(max(0, min(1, float(row["y2"]))) * h)
    return x1, y1, x2, y2

def _caption_map(captions_jsonl: str | Path) -> Dict[str, Dict]:
    """
    Build a quick lookup: frame_path -> {caption, timestamp_sec}
    """
    cap = {}
    items = load_jsonl(captions_jsonl)
    for it in items:
        fp = str(it.get("frame_path", "")).strip()
        if not fp:
            continue
        cap[fp] = {
            "caption": it.get("caption", ""),
            "timestamp_sec": float(it.get("timestamp_sec", 0.0)),
        }
    return cap

def _wrap_text(img_w: int, text: str, max_chars_per_line: int = 70) -> List[str]:
    """
    Simple greedy line wrap so captions don't overflow.
    """
    words = text.split()
    lines, cur = [], []
    for w in words:
        cur.append(w)
        if len(" ".join(cur)) > max_chars_per_line:
            cur.pop()
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    # Try to scale lines with image width a bit
    if img_w < 600:
        # shorter lines on small images
        lines = _wrap_text(int(img_w), text, max_chars_per_line=max(40, int(max_chars_per_line * 0.7))) if max_chars_per_line > 45 else lines
    return lines[:3]  # keep it tight

def _compose_card(
    img: np.ndarray,
    caption: str,
    scene_label: str,
    scene_conf: float,
    objects_unique: List[str],
    timestamp_sec: float | None,
    frame_name: str,
) -> np.ndarray:
    """
    Stack an info strip (white) below the image with neatly wrapped text.
    """
    h, w = img.shape[:2]
    pad = 10
    line_h = 22
    max_lines_caption = 2

    # Prepare text lines
    cap_lines = _wrap_text(w, f"Caption: {caption}", max_chars_per_line=80) if caption else ["Caption: -"]
    cap_lines = cap_lines[:max_lines_caption]
    scene_line = f"Scene: {scene_label or '-'} ({scene_conf:.2f})"
    obj_line = f"Objects: {', '.join(objects_unique[:6]) if objects_unique else '-'}"
    ts = f"{timestamp_sec:.2f}s" if timestamp_sec is not None else "-"
    meta_line = f"Frame: {frame_name}  |  Time: {ts}"

    # Compute strip height
    lines = cap_lines + [scene_line, obj_line, meta_line]
    strip_h = pad + len(lines) * line_h + pad

    # Compose
    canvas = np.ones((h + strip_h, w, 3), dtype=np.uint8) * 255
    canvas[:h, :, :] = img

    y = h + pad + 16
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (10, y), FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y += line_h

    return canvas

def visualize_composites(
    cfg_path: str | Path,
    captions_jsonl: str | Path,
    num_samples: int = 6,
    min_conf: float = 0.35,
    out_dir: str | Path = "task2_preprocess_analysis/outputs/visuals",
    prefer_with_detections: bool = True,
):
    """
    Create N composite images that fuse detections + scene + caption.
    """
    cfg = read_yaml(cfg_path)
    objects_path = Path(cfg["objects_path"])
    scenes_path = Path(cfg["scenes_path"])
    ensure_dir(out_dir)

    # Load
    if not objects_path.exists():
        raise FileNotFoundError(f"Objects parquet not found: {objects_path}")
    if not scenes_path.exists():
        raise FileNotFoundError(f"Scenes parquet not found: {scenes_path}")

    df_obj = pd.read_parquet(objects_path)
    df_scn = pd.read_parquet(scenes_path)
    caps = _caption_map(captions_jsonl)

    # Select candidate frames
    if prefer_with_detections and not df_obj.empty:
        cand = df_obj[df_obj["confidence"] >= float(min_conf)]["frame_path"].drop_duplicates()
        if cand.empty:
            cand = df_scn["frame_path"].drop_duplicates()
    else:
        cand = df_scn["frame_path"].drop_duplicates()

    # Downsample to requested number
    cand = cand.reset_index(drop=True)
    if num_samples is None or int(num_samples) <= 0:
        frames = cand
    elif len(cand) > int(num_samples):
        frames = cand.sample(int(num_samples), random_state=42)
    else:
        frames = cand

    count_ok = 0
    for fp in tqdm(frames, desc="Rendering composites", total=len(frames)):
        fp = str(fp)
        im = cv2.imread(fp)
        if im is None:
            logger.warning("Could not read image: %s", fp)
            continue

        h, w = im.shape[:2]

        # Draw boxes for this frame
        sub = df_obj[(df_obj["frame_path"] == fp) & (df_obj["confidence"] >= float(min_conf))]
        for _, r in sub.iterrows():
            x1, y1, x2, y2 = _denorm_xyxy(r, w, h)
            color = (50, 200, 50)  # green
            cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
            label = str(r["label"])
            conf = float(r["confidence"])
            txt = f"{label} {conf:.2f}"
            # text background
            (tw, th), _ = cv2.getTextSize(txt, FONT, 0.5, 1)
            cv2.rectangle(im, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(im, txt, (x1 + 2, y1 - 4), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Scene & objects summary
        scn = df_scn[df_scn["frame_path"] == fp].head(1)
        scene_label = scn.get("scene_label_final", scn.get("scene_label")).iloc[0] if not scn.empty else ""
        scene_conf = float(scn.get("scene_conf_smoothed", scn.get("confidence")).iloc[0]) if not scn.empty else 0.0

        objects_unique = sorted(sub["label"].unique().tolist())

        # Caption & timestamp
        cap_rec = caps.get(fp, {})
        caption = cap_rec.get("caption", "")
        timestamp_sec = cap_rec.get("timestamp_sec", None)

        # Compose card
        card = _compose_card(
            im,
            caption=caption,
            scene_label=scene_label,
            scene_conf=scene_conf,
            objects_unique=objects_unique,
            timestamp_sec=timestamp_sec,
            frame_name=Path(fp).name,
        )

        # Save
        out_path = Path(out_dir) / f"card_{Path(fp).stem}.jpg"
        cv2.imwrite(str(out_path), card)
        logger.info("Saved composite → %s", out_path)
        count_ok += 1

    logger.info("Wrote %d composite images to %s", count_ok, out_dir)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create composite analysis cards (frame + boxes + caption + scene).")
    parser.add_argument("--cfg", required=True, help="Path to task2.yaml")
    parser.add_argument(
        "--captions", required=False,
        default="task1_video_captioning/data/processed/captions/captions.jsonl",
        help="Path to Task 1 captions.jsonl")
    parser.add_argument("--num", type=int, default=6, help="Number of cards to produce")
    parser.add_argument("--min_conf", type=float, default=0.35, help="Min confidence for showing boxes")
    parser.add_argument("--out_dir", default="task2_preprocess_analysis/outputs/visuals")
    args = parser.parse_args()

    visualize_composites(
        cfg_path=args.cfg,
        captions_jsonl=args.captions,
        num_samples=args.num,
        min_conf=args.min_conf,
        out_dir=args.out_dir,
    )

if __name__ == "__main__":
    main()
