"""
Task 2 — Object Detection with YOLOv8 (Ultralytics)

WHAT THIS FILE DOES
-------------------
- Loads the frame manifest from Task 1 (frames.csv).
- Runs YOLOv8 (nano) on CPU over the listed frames (batch-friendly).
- Writes a long-format detections table to Parquet:
    [frame_path, timestamp_sec, label, confidence, x1, y1, x2, y2, xcenter, ycenter]


"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from common.io_utils import ensure_dir, read_yaml, save_parquet, init_logger

logger = init_logger("task2.detect")


def _normalize_boxes(
    xyxy: pd.DataFrame,
    width: int,
    height: int,
) -> pd.DataFrame:
    """
    Normalize xyxy boxes to [0,1] range and add centers.
    Expects columns: x1, y1, x2, y2 (pixel coords).
    """
    df = xyxy.copy()
    df["x1"] = df["x1"] / max(1, width)
    df["x2"] = df["x2"] / max(1, width)
    df["y1"] = df["y1"] / max(1, height)
    df["y2"] = df["y2"] / max(1, height)
    df["xcenter"] = (df["x1"] + df["x2"]) / 2.0
    df["ycenter"] = (df["y1"] + df["y2"]) / 2.0
    return df


def detect_objects(
    frames_csv: str | Path,
    out_parquet: str | Path,
    model_name: str = "yolov8n.pt",
    device: str = "cpu",
    batch_size: int = 8,
    conf_threshold: float = 0.3,
    max_images: Optional[int] = None,
) -> Path:
    """
    Run YOLOv8 over frames listed in frames_csv and save a detections table.

    Returns:
        Path to the written Parquet file.
    """
    frames_csv = Path(frames_csv)
    out_parquet = Path(out_parquet)
    ensure_dir(out_parquet.parent)

    # Load frame manifest (produced by Task 1)
    df_frames = pd.read_csv(frames_csv)
    if max_images is not None:
        df_frames = df_frames.head(max_images)

    if df_frames.empty:
        logger.warning("No frames found in %s", frames_csv)
        # write empty schema-compatible file
        empty = pd.DataFrame(
            columns=[
                "frame_path",
                "timestamp_sec",
                "label",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "xcenter",
                "ycenter",
            ]
        )
        save_parquet(empty, out_parquet)
        return out_parquet

    paths: List[str] = df_frames["frame_path"].astype(str).tolist()
    times: List[float] = df_frames["timestamp_sec"].astype(float).tolist()

    logger.info("Loaded %d frames from %s", len(paths), frames_csv)
    logger.info("Loading YOLO model: %s (device=%s)", model_name, device)
    model = YOLO(model_name)

    # Ultralytics .predict can take a list of paths; we'll batch manually for control.
    all_rows: List[Dict[str, Any]] = []
    names = model.model.names if hasattr(model, "model") else model.names

    for i in tqdm(range(0, len(paths), batch_size), desc="YOLO detecting"):
        batch_paths = paths[i : i + batch_size]
        batch_ts = times[i : i + batch_size]

        results = model.predict(
            source=batch_paths,
            conf=conf_threshold,
            device=0 if device == "cuda" else "cpu",
            verbose=False,
        )

        # Parse results per image
        for p, ts, res in zip(batch_paths, batch_ts, results):
            # res.boxes.xyxy: (n, 4) tensor; res.boxes.conf: (n,) ; res.boxes.cls: (n,)
            n_h, n_w = res.orig_shape  # (height, width)
            if res.boxes is None or len(res.boxes) == 0:
                continue

            b = res.boxes
            xyxy = b.xyxy.cpu().numpy()
            conf = b.conf.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)

            det_df = pd.DataFrame(
                xyxy, columns=["x1", "y1", "x2", "y2"]
            )
            det_df["confidence"] = conf
            det_df["label"] = [names[c] if c in names else str(c) for c in cls]

            det_df = _normalize_boxes(det_df, width=n_w, height=n_h)

            # attach frame/timestamp
            det_df.insert(0, "timestamp_sec", ts)
            det_df.insert(0, "frame_path", p)

            # Keep only columns we care about (stable schema)
            det_df = det_df[
                [
                    "frame_path",
                    "timestamp_sec",
                    "label",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "xcenter",
                    "ycenter",
                ]
            ]
            all_rows.append(det_df)

    if all_rows:
        out_df = pd.concat(all_rows, ignore_index=True)
    else:
        logger.warning("No detections above conf=%.2f", conf_threshold)
        out_df = pd.DataFrame(
            columns=[
                "frame_path",
                "timestamp_sec",
                "label",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "xcenter",
                "ycenter",
            ]
        )

    save_parquet(out_df, out_parquet)
    logger.info("Wrote detections → %s (rows=%d)", out_parquet, len(out_df))
    return out_parquet


# -----------------------
# Tiny CLI
# -----------------------

def main():

    import argparse

    parser = argparse.ArgumentParser(description="Task 2: Object detection over frames (YOLOv8n).")
    parser.add_argument("--cfg", required=True, help="Path to task2.yaml")
    parser.add_argument("--max_images", type=int, default=None, help="Limit frames for a quick run.")
    args = parser.parse_args()

    cfg = read_yaml(args.cfg)
    detect_objects(
        frames_csv=cfg["frames_csv"],
        out_parquet=cfg["objects_path"],
        model_name=cfg.get("object_model", "yolov8n.pt"),
        device=cfg.get("device", "cpu"),
        batch_size=int(cfg.get("batch_size", 8)),
        conf_threshold=float(cfg.get("conf_threshold", 0.3)),
        max_images=args.max_images or cfg.get("max_images", None),
    )


if __name__ == "__main__":
    main()
