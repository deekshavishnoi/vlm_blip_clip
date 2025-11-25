"""
Task 3 — Build a unified per-frame knowledge base from Tasks 1 & 2

WHAT THIS FILE DOES
-------------------
- Loads:
    - captions.jsonl  (Task 1; per-frame captions with timestamps)
    - objects.parquet (Task 2; YOLO detections)
    - scenes.parquet  (Task 2; Places365 top-1 scene)
- Aggregates to ONE ROW PER FRAME with these columns:
    - frame_path: str
    - timestamp_sec: float
    - caption: str
    - objects: List[str]            (unique object labels on the frame)
    - objects_top: List[str]        (top-N objects by frequency/confidence)
    - objects_str: str              (comma-joined for embedding)
    - scene_label: str
    - scene_confidence: float
    - description: str              (caption + objects + scene; for embedding)

- Writes a compact Parquet table for Task 3:
    knowledge_base.parquet

"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from common.io_utils import read_yaml, ensure_dir, load_jsonl, save_parquet, init_logger

logger = init_logger("task3.kb")


def _load_captions(captions_jsonl: str | Path) -> pd.DataFrame:
    """Load Task 1 captions.jsonl → DataFrame [frame_path, timestamp_sec, caption]."""
    items = load_jsonl(captions_jsonl)
    if not items:
        logger.warning("No captions found at %s", captions_jsonl)
        return pd.DataFrame(columns=["frame_path", "timestamp_sec", "caption"])
    df = pd.DataFrame(items)
    # keep stable columns if extra keys exist
    cols = [c for c in ["frame_path", "timestamp_sec", "caption"] if c in df.columns]
    return df[cols].drop_duplicates()


def _load_objects(objects_parquet: str | Path) -> pd.DataFrame:
    """Load Task 2 objects.parquet → DataFrame; may be empty."""
    p = Path(objects_parquet)
    if not p.exists():
        logger.warning("Objects parquet not found: %s", p)
        return pd.DataFrame(columns=["frame_path", "label", "confidence"])
    df = pd.read_parquet(p)
    needed = ["frame_path", "label", "confidence"]
    for n in needed:
        if n not in df.columns:
            df[n] = [] if n == "label" else 0.0
    return df[needed]


def _load_scenes(scenes_parquet: str | Path) -> pd.DataFrame:
    """Load Task 2 scenes.parquet → DataFrame [frame_path, scene_label, confidence, timestamp_sec?]."""
    p = Path(scenes_parquet)
    if not p.exists():
        logger.warning("Scenes parquet not found: %s", p)
        return pd.DataFrame(columns=["frame_path", "scene_label", "confidence"])
    df = pd.read_parquet(p)
    needed = ["frame_path", "scene_label", "confidence"]
    for n in needed:
        if n not in df.columns:
            df[n] = "" if n == "scene_label" else 0.0
    return df[needed].drop_duplicates("frame_path")


def _aggregate_objects(
    df_obj: pd.DataFrame,
    min_conf: float = 0.25,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Collapse many detections → per-frame lists and top-N summary.
    Returns DataFrame with:
        frame_path, objects (list), objects_top (list), objects_str (str)
    """
    if df_obj.empty:
        return pd.DataFrame(columns=["frame_path", "objects", "objects_top", "objects_str"])

    # filter by confidence
    df = df_obj[df_obj["confidence"] >= float(min_conf)].copy()
    if df.empty:
        return pd.DataFrame(columns=["frame_path", "objects", "objects_top", "objects_str"])

    # all unique labels on frame
    all_objs = df.groupby("frame_path")["label"].apply(lambda s: sorted(set(s))).rename("objects")

    # top-N by (count, mean confidence)
    by_frame = []
    for fp, sub in df.groupby("frame_path"):
        counts = sub.groupby("label").agg(
            count=("label", "size"),
            mean_conf=("confidence", "mean"),
        ).sort_values(["count", "mean_conf"], ascending=[False, False])
        top_labels = counts.head(top_n).index.tolist()
        by_frame.append({"frame_path": fp, "objects_top": top_labels})
    top_df = pd.DataFrame(by_frame)

    out = all_objs.reset_index().merge(top_df, on="frame_path", how="left")
    out["objects_top"] = out["objects_top"].apply(lambda x: x if isinstance(x, list) else [])
    out["objects_str"] = out["objects"].apply(lambda xs: ", ".join(xs) if isinstance(xs, list) else "")
    return out


def _compose_description(caption: str, objects_str: str, scene_label: str) -> str:
    """Make a short natural string to embed/search later."""
    parts: List[str] = []
    if caption:
        parts.append(caption.strip())
    if objects_str:
        parts.append(f"Objects: {objects_str}.")
    if scene_label:
        parts.append(f"Scene: {scene_label}.")
    text = " ".join(parts).strip()
    return text or "No description."


def build_knowledge_base(
    captions_path: str | Path,
    objects_path: str | Path,
    scenes_path: str | Path,
    out_parquet: str | Path,
    min_obj_conf: float = 0.25,
    top_n_objects: int = 5,
) -> Path:
    """
    Merge captions+objects+scenes → one row per frame with a rich 'description'.
    """
    captions = _load_captions(captions_path)
    objects = _load_objects(objects_path)
    scenes = _load_scenes(scenes_path)

    if captions.empty:
        logger.warning("Captions table is empty; knowledge base will be minimal.")
    logger.info("Loaded: captions=%d rows, objects=%d rows, scenes=%d rows",
                len(captions), len(objects), len(scenes))

    # Aggregate objects per frame
    agg_obj = _aggregate_objects(objects, min_conf=min_obj_conf, top_n=top_n_objects)

    # Merge all on frame_path (left join on captions so we keep the same frames Task 1 used)
    kb = captions.merge(agg_obj, on="frame_path", how="left").merge(
        scenes, on="frame_path", how="left"
    )

    # --- FIX: normalize scene confidence column name ---
    if "confidence" in kb.columns:
        kb = kb.rename(columns={"confidence": "scene_confidence"})
    if "scene_confidence" not in kb.columns:
        kb["scene_confidence"] = 0.0

    # Fill NaNs
    kb["caption"] = kb["caption"].fillna("")
    kb["objects"] = kb["objects"].apply(lambda x: x if isinstance(x, list) else [])
    kb["objects_top"] = kb["objects_top"].apply(lambda x: x if isinstance(x, list) else [])
    kb["objects_str"] = kb["objects_str"].fillna("")
    kb["scene_label"] = kb["scene_label"].fillna("")
    kb["scene_confidence"] = kb["scene_confidence"].fillna(0.0)

    # Compose final free-text description
    kb["description"] = kb.apply(
        lambda r: _compose_description(r.get("caption", ""), r.get("objects_str", ""), r.get("scene_label", "")),
        axis=1,
    )

    # Reorder columns
    cols = [
        "frame_path",
        "timestamp_sec",
        "caption",
        "objects",
        "objects_top",
        "objects_str",
        "scene_label",
        "scene_confidence",
        "description",
    ]
    kb = kb[cols].sort_values("timestamp_sec").reset_index(drop=True)

    # Write
    out_parquet = Path(out_parquet)
    ensure_dir(out_parquet.parent)
    save_parquet(kb, out_parquet)
    logger.info("Wrote knowledge base → %s (rows=%d)", out_parquet, len(kb))
    return out_parquet


# -----------------------
# CLI
# -----------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build per-frame knowledge base for Task 3.")
    parser.add_argument("--cfg", required=True, help="Path to task3.yaml")
    parser.add_argument("--top_n_objects", type=int, default=5)
    parser.add_argument("--min_obj_conf", type=float, default=0.25)
    args = parser.parse_args()

    cfg = read_yaml(args.cfg)
    build_knowledge_base(
        captions_path=cfg["captions_path"],
        objects_path=cfg["objects_path"],
        scenes_path=cfg["scenes_path"],
        out_parquet=cfg["knowledge_base_path"],
        min_obj_conf=args.min_obj_conf,
        top_n_objects=args.top_n_objects,
    )


if __name__ == "__main__":
    main()
