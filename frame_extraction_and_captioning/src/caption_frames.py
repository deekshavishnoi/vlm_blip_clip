"""
Task 1 — Per-frame captioning (Hugging Face Transformers)

WHAT THIS FILE DOES
-------------------
- Loads a lightweight image-captioning model (BLIP by default).
- Reads the frame manifest (CSV) produced by extract_frames.py.
- Iterates over frames and generates captions.
- Saves results to JSONL with useful metadata for later analysis.

"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional
import time

import pandas as pd
from PIL import Image
from transformers import pipeline

from common.io_utils import ensure_dir, read_yaml, save_jsonl, init_logger

logger = init_logger("task1.caption")


def _load_images(paths: Iterable[str | Path]) -> List[Image.Image]:
    """Loading images into memory (PIL)"""
    imgs: List[Image.Image] = []
    for p in paths:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB"))
    return imgs


def caption_frames(
    frames_csv: str | Path,
    out_jsonl: str | Path,
    model_name: str = "Salesforce/blip-image-captioning-base",
    device: str = "cpu",
    batch_size: int = 4,
    max_frames: Optional[int] = None,
) -> None:
    """
    Run captioning on frames listed in frames_csv and write JSONL results.

    Args:
        frames_csv: Path to CSV with columns [frame_path, timestamp_sec, ...]
        out_jsonl: Where to write JSONL with captions.
        model_name: HF model id (BLIP by default).
        device: "cpu" or "cuda" (we force CPU-friendly defaults).
        batch_size: Small batches to amortize model overhead on CPU.
        max_frames: Optional hard cap for debugging/speed.

    Returns:
        None (writes JSONL to disk).
    """
    frames_csv = Path(frames_csv)
    out_jsonl = Path(out_jsonl)
    ensure_dir(out_jsonl.parent)

    # Load manifest
    df = pd.read_csv(frames_csv)
    if max_frames is not None:
        df = df.head(max_frames)
    logger.info("Loaded %d frame entries from %s", len(df), frames_csv)

    # Initialize captioning pipeline
    # Note: task="image-to-text" is the correct alias across transformers versions.
    captioner = pipeline(
        task="image-to-text",
        model=model_name,
        device=-1 if device == "cpu" else 0,
    )
    logger.info("Loaded model: %s (device=%s)", model_name, device)

    records: List[Dict[str, Any]] = []

    # Process in small batches for efficiency on CPU
    paths = df["frame_path"].tolist()
    timestamps = df["timestamp_sec"].tolist()

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        batch_ts = timestamps[i : i + batch_size]

        images = _load_images(batch_paths)

        t0 = time.time()
        outputs = captioner(images)  # returns list of [{"generated_text": "..."}]
        dt_ms = (time.time() - t0) * 1000.0

        # Distribute batch time roughly per item
        per_item_ms = dt_ms / max(1, len(images))

        for p, ts, out in zip(batch_paths, batch_ts, outputs):
            caption = out[0]["generated_text"] if isinstance(out, list) else out["generated_text"]
            records.append(
                {
                    "frame_path": str(p),
                    "timestamp_sec": float(ts),
                    "caption": caption,
                    "model": model_name,
                    "runtime_ms": round(per_item_ms, 2),
                }
            )

        logger.debug("Processed frames %d..%d in ~%.1f ms", i, min(i + batch_size, len(paths)) - 1, dt_ms)

    # Write JSONL
    save_jsonl(records, out_jsonl)
    logger.info("Wrote %d captions → %s", len(records), out_jsonl)


# -----------------------
# Convenience CLI
# -----------------------

def main():

    import argparse
    parser = argparse.ArgumentParser(description="Caption frames with a lightweight VLM (BLIP).")
    parser.add_argument("--cfg", required=True, help="Path to YAML config (from Task 1).")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit frames for quick tests.")
    args = parser.parse_args()

    cfg = read_yaml(args.cfg)
    caption_frames(
        frames_csv=cfg["frames_csv_path"],
        out_jsonl=cfg["captions_path"],
        model_name=cfg.get("caption_model", "Salesforce/blip-image-captioning-base"),
        device=cfg.get("device", "cpu"),
        batch_size=cfg.get("batch_size", 4),
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
