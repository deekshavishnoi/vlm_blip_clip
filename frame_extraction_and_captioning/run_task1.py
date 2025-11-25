"""
Task 1 — End-to-end runner (CLI)

WHAT THIS FILE DOES
-------------------
Provides a simple command-line interface to:
  - extract  : turn video into frames (+ frames.csv)
  - caption  : run BLIP on frames → captions.jsonl
  - summarize: compress per-frame captions → description.txt
  - all      : run the whole pipeline in sequence


USAGE
-----
# Run everything with defaults from YAML
python task1_video_captioning/run_task1.py all --cfg task1_video_captioning/configs/task1.yaml

# Or run step-by-step:
python task1_video_captioning/run_task1.py extract  --cfg task1_video_captioning/configs/task1.yaml
python task1_video_captioning/run_task1.py caption  --cfg task1_video_captioning/configs/task1.yaml
python task1_video_captioning/run_task1.py summarize --cfg task1_video_captioning/configs/task1.yaml

"""

from __future__ import annotations
import argparse
from pathlib import Path


from common.io_utils import read_yaml, ensure_dir, init_logger
from task1_video_captioning.src.extract_frames import extract_frames
from task1_video_captioning.src.caption_frames import caption_frames
from task1_video_captioning.src.summarize import summarize_captions
from task1_video_captioning.src.caption_with_rerank import caption_frames_with_rerank


logger = init_logger("task1.runner")


def step_extract(cfg: dict, override_max_frames: int | None = None) -> Path:
    """Extract frames from the configured video and save frames.csv."""
    video_path = cfg["video_path"]
    out_dir = cfg["output_frames_dir"]
    frames_csv = cfg["frames_csv_path"]
    fps = float(cfg.get("frames_per_second", 1.0))
    img_ext = cfg.get("image_ext", ".jpg")
    max_frames = override_max_frames if override_max_frames is not None else cfg.get("max_frames", None)

    ensure_dir(out_dir)
    df = extract_frames(
        video_path=video_path,
        output_dir=out_dir,
        frames_per_second=fps,
        image_ext=img_ext,
        max_frames=max_frames,
    )
    Path(frames_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(frames_csv, index=False)
    logger.info("Extract step complete → %s (frames=%d)", frames_csv, len(df))
    return Path(frames_csv)


def step_caption(cfg: dict, override_max_frames: int | None = None) -> Path:
    """Caption frames listed in frames.csv and write captions.jsonl."""
    frames_csv = cfg["frames_csv_path"]
    out_jsonl = cfg["captions_path"]
    model_name = cfg.get("caption_model", "Salesforce/blip-image-captioning-base")
    device = cfg.get("device", "cpu")
    batch_size = int(cfg.get("batch_size", 4))
    max_frames = override_max_frames if override_max_frames is not None else cfg.get("max_frames", None)

    caption_frames(
        frames_csv=frames_csv,
        out_jsonl=out_jsonl,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        max_frames=max_frames,
    )
    logger.info("Caption step complete → %s", out_jsonl)
    return Path(out_jsonl)

def step_caption_rerank(cfg: dict, override_max_frames: int | None = None) -> Path:
    """
    Caption frames using BLIP (beam search) + CLIP re-ranking.
    Writes captions.jsonl and optional caption cards (images) for README.
    """
    frames_csv = cfg["frames_csv_path"]
    out_jsonl = cfg["captions_path"]

    # optional config keys with sensible defaults
    cards_dir = cfg.get("caption_cards_dir", None)
    device = cfg.get("device", "cpu")
    num_beams = int(cfg.get("num_beams", 5))
    num_return = int(cfg.get("num_return", 5))
    max_new_tokens = int(cfg.get("max_new_tokens", 30))
    max_images = override_max_frames if override_max_frames is not None else cfg.get("max_frames", None)

    caption_frames_with_rerank(
        frames_csv=frames_csv,
        out_jsonl=out_jsonl,
        cards_dir=cards_dir,
        device=device,
        num_beams=num_beams,
        num_return=num_return,
        max_new_tokens=max_new_tokens,
        max_images=max_images,
    )
    logger.info("Caption (rerank) step complete → %s", out_jsonl)
    return Path(out_jsonl)


def step_summarize(cfg: dict) -> Path:
    """Summarize per-frame captions into a short video-level description."""
    captions_jsonl = cfg["captions_path"]
    out_txt = cfg["video_description_path"]
    dedupe_thr = float(cfg.get("dedupe_thr", 0.8))
    max_sentences = int(cfg.get("max_sentences", 5))

    paragraph = summarize_captions(
        captions_jsonl=captions_jsonl,
        out_txt=out_txt,
        dedupe_threshold=dedupe_thr,
        max_sentences=max_sentences,
    )
    logger.info("Summarize step complete → %s", out_txt)
    logger.info("Preview: %s", paragraph[:140] + ("..." if len(paragraph) > 140 else ""))
    return Path(out_txt)


def main():
    parser = argparse.ArgumentParser(description="Task 1 runner: extract → caption → summarize")
    sub = parser.add_subparsers(dest="cmd", required=True)



    # shared flag
    def add_shared(p):
        p.add_argument("--cfg", required=True, help="Path to YAML config for Task 1.")
        p.add_argument("--max_frames", type=int, default=None, help="Optional cap for quick tests.")



    p_ext = sub.add_parser("extract", help="Extract frames from video")
    add_shared(p_ext)

    p_cap = sub.add_parser("caption", help="Run per-frame captioning")
    add_shared(p_cap)

    p_cap_rr = sub.add_parser("caption_rerank", help="Run BLIP + CLIP re-ranking for better captions")
    add_shared(p_cap_rr)

    p_sum = sub.add_parser("summarize", help="Summarize captions into a video description")
    p_sum.add_argument("--cfg", required=True)

    p_all = sub.add_parser("all", help="Run extract → caption → summarize")
    add_shared(p_all)

    args = parser.parse_args()
    cfg = read_yaml(args.cfg)

    if args.cmd == "extract":
        step_extract(cfg, override_max_frames=args.max_frames)

    elif args.cmd == "caption":
        # Ensure frames exist (basic check)
        if not Path(cfg["frames_csv_path"]).exists():
            logger.warning("frames_csv_path not found. Running extract first...")
            step_extract(cfg, override_max_frames=args.max_frames)
        step_caption(cfg, override_max_frames=args.max_frames)

    elif args.cmd == "summarize":
        # Ensure captions exist (basic check)
        if not Path(cfg["captions_path"]).exists():
            raise FileNotFoundError("captions_path not found. Run 'caption' first.")
        step_summarize(cfg)

    elif args.cmd == "caption_rerank":
        # Ensure frames exist (basic check)
        if not Path(cfg["frames_csv_path"]).exists():
            logger.warning("frames_csv_path not found. Running extract first...")
            step_extract(cfg, override_max_frames=args.max_frames)
        step_caption_rerank(cfg, override_max_frames=args.max_frames)



    elif args.cmd == "all":

        step_extract(cfg, override_max_frames=args.max_frames)
        # Old:
        # step_caption(cfg, override_max_frames=args.max_frames)
        # New:
        step_caption_rerank(cfg, override_max_frames=args.max_frames)
        step_summarize(cfg)

    logger.info("Done.")


if __name__ == "__main__":
    main()
