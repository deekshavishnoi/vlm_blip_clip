"""
Task 2 — Summarize & Verify Analysis Outputs

WHAT THIS FILE DOES
-------------------
- Loads Task 2 outputs:
    - objects.parquet  (YOLO detections)
    - scenes.parquet   (Places365 scene predictions)
- Prints human-readable summaries (top classes, counts, basic stats).
- Verifies expected columns / non-emptiness and reports issues.
- Saves quick plots (top objects / scenes).

"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict

import pandas as pd

from common.io_utils import read_yaml, ensure_dir, init_logger
from common.viz import save_bar_chart, draw_boxes_on_image

logger = init_logger("task2.summarize")


# -----------------------
# Verification helpers
# -----------------------

def _verify_objects_schema(df: pd.DataFrame) -> List[str]:
    required = ["frame_path", "timestamp_sec", "label", "confidence", "x1", "y1", "x2", "y2"]
    missing = [c for c in required if c not in df.columns]
    return missing


def _verify_scenes_schema(df: pd.DataFrame) -> List[str]:
    required = ["frame_path", "timestamp_sec", "scene_label", "confidence"]
    missing = [c for c in required if c not in df.columns]
    return missing


# -----------------------
# Summaries
# -----------------------

def summarize_objects(objects_parquet: str | Path, plots_dir: str | Path, top_k: int = 5) -> pd.DataFrame:
    p = Path(objects_parquet)
    if not p.exists():
        logger.error("Objects file not found: %s", p)
        return pd.DataFrame()

    df = pd.read_parquet(p)
    if df.empty:
        logger.warning("Objects parquet is empty.")
        return df

    # Schema check
    missing = _verify_objects_schema(df)
    if missing:
        logger.warning("Objects parquet missing columns: %s", missing)

    # Text summary
    logger.info("Object detections: %d rows, %d unique labels", len(df), df["label"].nunique())
    logger.info("Top labels:\n%s", df["label"].value_counts().head(top_k).to_string())
    logger.info("Mean confidence per label (top %d):\n%s",
                top_k, df.groupby("label")["confidence"].mean().sort_values(ascending=False).head(top_k).round(3).to_string())

    # Plot top objects
    counts = df["label"].value_counts().head(top_k)
    ensure_dir(plots_dir)
    save_bar_chart(
        labels=counts.index.tolist(),
        counts=counts.values.tolist(),
        title=f"Top-{top_k} detected objects",
        out_path=Path(plots_dir) / "top_objects.png",
    )
    logger.info("Saved plot: %s", Path(plots_dir) / "top_objects.png")
    return df


def summarize_scenes(scenes_parquet: str | Path, plots_dir: str | Path, top_k: int = 5) -> pd.DataFrame:
    p = Path(scenes_parquet)
    if not p.exists():
        logger.error("Scenes file not found: %s", p)
        return pd.DataFrame()

    df = pd.read_parquet(p)
    if df.empty:
        logger.warning("Scenes parquet is empty.")
        return df

    # Schema check
    missing = _verify_scenes_schema(df)
    if missing:
        logger.warning("Scenes parquet missing columns: %s", missing)

    # Text summary
    logger.info("Scene classifications: %d rows, %d unique scenes", len(df), df["scene_label"].nunique())
    logger.info("Top scenes:\n%s", df["scene_label"].value_counts().head(top_k).to_string())
    logger.info("Mean scene confidence (top %d):\n%s",
                top_k, df.groupby("scene_label")["confidence"].mean().sort_values(ascending=False).head(top_k).round(3).to_string())

    # Plot top scenes
    counts = df["scene_label"].value_counts().head(top_k)
    ensure_dir(plots_dir)
    save_bar_chart(
        labels=counts.index.tolist(),
        counts=counts.values.tolist(),
        title=f"Top-{top_k} scenes",
        out_path=Path(plots_dir) / "top_scenes.png",
    )
    logger.info("Saved plot: %s", Path(plots_dir) / "top_scenes.png")
    return df


# -----------------------
# Visual sanity checks (optional)
# -----------------------

def make_sample_overlays(
    df_objects: pd.DataFrame,
    samples_dir: str | Path,
    max_images: int = 6,
    min_conf: float = 0.35,
) -> None:
    """
    Create a few annotated frames with detection boxes.
    Uses normalized coords (x1..y2 in [0,1]) written by detect_objects.py.
    """
    if df_objects.empty:
        logger.info("No objects to overlay; skipping.")
        return

    ensure_dir(samples_dir)

    # Choose up to N distinct frames that have at least one detection over min_conf
    df_filt = df_objects[df_objects["confidence"] >= min_conf]
    if df_filt.empty:
        logger.info("No detections above min_conf=%.2f to overlay.", min_conf)
        return

    frames = df_filt["frame_path"].drop_duplicates().head(max_images).tolist()
    for fp in frames:
        sub = df_filt[df_filt["frame_path"] == fp]
        boxes = [
            {
                "x1": float(r["x1"]), "y1": float(r["y1"]),
                "x2": float(r["x2"]), "y2": float(r["y2"]),
                "label": str(r["label"]),
                "confidence": float(r["confidence"]),
            }
            for _, r in sub.iterrows()
        ]
        out_path = Path(samples_dir) / (Path(fp).stem + "_boxes.jpg")
        try:
            draw_boxes_on_image(fp, boxes, out_path, labels_key="label", xyxy_norm=True)
            logger.info("Saved overlay: %s", out_path)
        except Exception as e:
            logger.warning("Failed to draw overlay for %s: %s", fp, e)


# -----------------------
# CLI
# -----------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Summarize & verify Task 2 parquet outputs.")
    parser.add_argument("--cfg", required=True, help="Path to task2.yaml")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k items for summary tables/plots.")
    parser.add_argument("--num_overlay", type=int, default=6, help="How many frames to save with detection overlays.")
    parser.add_argument("--min_conf", type=float, default=0.35, help="Min confidence for drawing boxes.")
    args = parser.parse_args()

    cfg = read_yaml(args.cfg)
    objects_path = cfg["objects_path"]
    scenes_path = cfg["scenes_path"]
    plots_dir = cfg.get("plots_dir", "task2_preprocess_analysis/outputs/plots")
    samples_dir = Path(plots_dir).parent / "samples"

    logger.info("Verifying and summarizing parquet outputs...")
    df_obj = summarize_objects(objects_path, plots_dir, top_k=args.top_k)
    df_scn = summarize_scenes(scenes_path, plots_dir, top_k=args.top_k)

    # Optional overlays for a few frames with confident detections
    make_sample_overlays(df_obj, samples_dir=samples_dir, max_images=args.num_overlay, min_conf=args.min_conf)

    # Basic cross-check: ensure both files cover roughly same set of frames
    try:
        if not df_obj.empty and not df_scn.empty:
            f_obj = set(df_obj["frame_path"].unique())
            f_scn = set(df_scn["frame_path"].unique())
            inter = len(f_obj & f_scn)
            logger.info("Frames with detections: %d | with scenes: %d | in both: %d", len(f_obj), len(f_scn), inter)
    except Exception:
        pass

    logger.info("Summarize & verify finished.")


if __name__ == "__main__":
    main()
