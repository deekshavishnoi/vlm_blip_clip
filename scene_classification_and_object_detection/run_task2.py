"""
Task 2 — End-to-end runner (CLI)

WHAT THIS FILE DOES
-------------------
Provides a simple CLI to:
  - detect   : run YOLOv8n object detection → objects.parquet
  - scenes   : run Places365 scene classification → scenes.parquet
  - all      : run detect → scenes
  - summarize: (optional) print quick top-k summaries to console

USAGE
-----
python -m task2_preprocess_analysis.run_task2 all \
    --cfg task2_preprocess_analysis/configs/task2.yaml

python -m task2_preprocess_analysis.run_task2 detect --cfg ...
python -m task2_preprocess_analysis.run_task2 scenes --cfg ...
python -m task2_preprocess_analysis.run_task2 summarize --cfg ...
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from common.io_utils import read_yaml, init_logger, ensure_dir
from task2_preprocess_analysis.src.detect_objects import detect_objects
from task2_preprocess_analysis.src.classify_scenes import classify_scenes
from task2_preprocess_analysis.src.visualize_results import visualize_composites
from common.viz import save_bar_chart

logger = init_logger("task2.runner")


def _summarize_objects(objects_parquet: str | Path, plots_dir: str | Path, top_k: int = 5) -> None:
    p = Path(objects_parquet)
    if not p.exists():
        logger.warning("Objects file not found: %s", p)
        return
    df = pd.read_parquet(p)
    if df.empty:
        logger.info("No object detections to summarize.")
        return
    counts = df["label"].value_counts().head(top_k)
    ensure_dir(plots_dir)
    save_bar_chart(
        labels=counts.index.tolist(),
        counts=counts.values.tolist(),
        title=f"Top-{top_k} detected objects",
        out_path=Path(plots_dir) / "top_objects.png",
    )
    logger.info("Top objects:\n%s", counts.to_string())


def _summarize_scenes(scenes_parquet: str | Path, plots_dir: str | Path, top_k: int = 5) -> None:
    p = Path(scenes_parquet)
    if not p.exists():
        logger.warning("Scenes file not found: %s", p)
        return
    df = pd.read_parquet(p)
    if df.empty:
        logger.info("No scene predictions to summarize.")
        return
    counts = df["scene_label"].value_counts().head(top_k)
    ensure_dir(plots_dir)
    save_bar_chart(
        labels=counts.index.tolist(),
        counts=counts.values.tolist(),
        title=f"Top-{top_k} scenes",
        out_path=Path(plots_dir) / "top_scenes.png",
    )
    logger.info("Top scenes:\n%s", counts.to_string())


def main():
    parser = argparse.ArgumentParser(description="Task 2 runner: detect → scenes → summarize")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_shared(p):
        p.add_argument("--cfg", required=True, help="Path to YAML config for Task 2.")
        p.add_argument("--max_images", type=int, default=None, help="Optional cap for quick tests.")

    p_det = sub.add_parser("detect", help="Run YOLO object detection")
    add_shared(p_det)

    p_scn = sub.add_parser("scenes", help="Run Places365 scene classification")
    add_shared(p_scn)

    p_sum = sub.add_parser("summarize", help="Save quick summary plots")
    p_sum.add_argument("--cfg", required=True)

    p_viz = sub.add_parser("viz", help="Generate annotated frame composites")
    add_shared(p_viz)
    p_viz.add_argument("--num", type=int, default=-1, help="Number of frames to visualize (-1 = all)")
    p_viz.add_argument("--min_conf", type=float, default=0.35, help="Confidence for drawing boxes")
    p_viz.add_argument("--captions", default="task1_video_captioning/data/processed/captions/captions.jsonl")
    p_viz.add_argument("--out_dir", default="task2_preprocess_analysis/outputs/visuals_all")
    p_viz.add_argument("--prefer_with_detections", action="store_true", help="Prefer frames that have detections")

    p_all = sub.add_parser("all", help="Run detect → scenes")
    add_shared(p_all)

    args = parser.parse_args()
    cfg = read_yaml(args.cfg)

    if args.cmd == "detect":
        detect_objects(
            frames_csv=cfg["frames_csv"],
            out_parquet=cfg["objects_path"],
            model_name=cfg.get("object_model", "yolov8n.pt"),
            device=cfg.get("device", "cpu"),
            batch_size=int(cfg.get("batch_size", 8)),
            conf_threshold=float(cfg.get("conf_threshold", 0.3)),
            max_images=args.max_images or cfg.get("max_images", None),
        )

    elif args.cmd == "scenes":
        classify_scenes(
            frames_csv=cfg["frames_csv"],
            out_parquet=cfg["scenes_path"],
            use_places365=True,
            max_images=args.max_images or cfg.get("max_images", None),
        )

    elif args.cmd == "summarize":
        _summarize_objects(cfg["objects_path"], cfg.get("plots_dir", "task2_preprocess_analysis/outputs/plots"))
        _summarize_scenes(cfg["scenes_path"], cfg.get("plots_dir", "task2_preprocess_analysis/outputs/plots"))


    elif args.cmd == "viz":
        visualize_composites(
            cfg_path=args.cfg,
            captions_jsonl=args.captions,
            num_samples=args.num,
            min_conf=args.min_conf,
            out_dir=args.out_dir,
            prefer_with_detections=args.prefer_with_detections,
        )

    elif args.cmd == "all":
        # run both steps
        detect_objects(
            frames_csv=cfg["frames_csv"],
            out_parquet=cfg["objects_path"],
            model_name=cfg.get("object_model", "yolov8n.pt"),
            device=cfg.get("device", "cpu"),
            batch_size=int(cfg.get("batch_size", 8)),
            conf_threshold=float(cfg.get("conf_threshold", 0.3)),
            max_images=args.max_images or cfg.get("max_images", None),
        )
        classify_scenes(
            frames_csv=cfg["frames_csv"],
            out_parquet=cfg["scenes_path"],
            use_places365=True,
            max_images=args.max_images or cfg.get("max_images", None),
        )
        visualize_composites(
            cfg_path=args.cfg,
            captions_jsonl="task1_video_captioning/data/processed/captions/captions.jsonl",
            num_samples=-1,
            min_conf=float(cfg.get("conf_threshold", 0.35)),
            out_dir="task2_preprocess_analysis/outputs/visuals_all",
            prefer_with_detections=True,
        )

        # optional quick plots for README
        _summarize_objects(cfg["objects_path"], cfg.get("plots_dir", "task2_preprocess_analysis/outputs/plots"))
        _summarize_scenes(cfg["scenes_path"], cfg.get("plots_dir", "task2_preprocess_analysis/outputs/plots"))

    logger.info("Task 2 finished.")


if __name__ == "__main__":
    main()
