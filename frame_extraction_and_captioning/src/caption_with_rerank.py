"""
Task 1 — Captioning with safer decoding + CLIP re-ranking
- Uses BLIP ("Salesforce/blip-image-captioning-base") with beam search to generate N candidates.
- Re-ranks candidates using CLIP image–text similarity.
- Writes per-frame captions to JSONL and optional caption cards (image + caption below).

USAGE (from project root)
-------------------------
python -m task1_video_captioning.src.caption_with_rerank \
  --frames_csv task1_video_captioning/data/interim/frames.csv \
  --out_jsonl task1_video_captioning/data/processed/captions/captions.jsonl \
  --cards_dir task1_video_captioning/outputs/caption_cards \
  --num_beams 5 --num_return 5 --max_new_tokens 30
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional

import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from transformers import pipeline, CLIPProcessor, CLIPModel

from common.io_utils import ensure_dir, init_logger

logger = init_logger("task1.caption_rerank")


# -------------------------------
# Models
# -------------------------------

def load_blip(device: str = "cpu"):
    """
    Safer decoding via beam search is passed at generation time.
    """
    cap = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
        device_map="cpu" if device == "cpu" else None,
    )
    return cap

def load_clip():
    """
    CLIP is used only for re-ranking a few short captions.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, proc

@torch.inference_mode()
def clip_scores(img: Image.Image, texts: List[str], model: CLIPModel, proc: CLIPProcessor) -> List[float]:
    inputs = proc(text=texts, images=img, return_tensors="pt", padding=True)
    out = model(**inputs)
    sims = out.logits_per_image.squeeze()  # higher is better
    return sims.tolist() if isinstance(sims.tolist(), list) else [float(sims.item())]


# -------------------------------
# Captioning + Re-rank
# -------------------------------

def caption_one_image(
    img_path: str | Path,
    blip_pipe,
    clip_model,
    clip_proc,
    num_beams: int = 5,
    num_return: int = 5,
    max_new_tokens: int = 30,
) -> str:
    """
    Generate N candidate captions with beam search, then pick the best via CLIP.
    """
    img = Image.open(img_path).convert("RGB")

    # Generate N candidates (no sampling, deterministic beam search)
    candidates = blip_pipe(
        img,
        generate_kwargs=dict(
            num_beams=num_beams,
            num_return_sequences=num_return,
            no_repeat_ngram_size=2,
            length_penalty=1.0,
            max_new_tokens=max_new_tokens,
        ),
    )
    texts = [c["generated_text"].strip() for c in candidates if c.get("generated_text")]

    # Fall back if BLIP returned nothing
    if not texts:
        return ""

    # Re-rank with CLIP
    scores = clip_scores(img, texts, clip_model, clip_proc)
    best_idx = int(torch.tensor(scores).argmax().item())
    return texts[best_idx]


# -------------------------------
# Card rendering (image + caption)
# -------------------------------

def render_caption_card(img_bgr: np.ndarray, caption: str, timestamp_sec: Optional[float], frame_name: str) -> np.ndarray:
    """
    Create a white strip under the image with caption + meta.
    """
    h, w = img_bgr.shape[:2]
    pad = 10
    line_h = 24
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Basic wrapping
    def wrap(text: str, max_chars: int = 80):
        words, lines, cur = text.split(), [], []
        for w_ in words:
            cur.append(w_)
            if len(" ".join(cur)) > max_chars:
                cur.pop()
                lines.append(" ".join(cur))
                cur = [w_]
        if cur:
            lines.append(" ".join(cur))
        return lines[:3]

    cap_lines = wrap(f"Caption: {caption}" if caption else "Caption: -")
    meta_line = f"Frame: {frame_name}  |  Time: {timestamp_sec:.2f}s" if timestamp_sec is not None else f"Frame: {frame_name}"

    lines = cap_lines + [meta_line]
    strip_h = pad + len(lines) * line_h + pad

    canvas = np.ones((h + strip_h, w, 3), dtype=np.uint8) * 255
    canvas[:h, :, :] = img_bgr

    y = h + pad + 18
    for line in lines:
        cv2.putText(canvas, line, (10, y), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y += line_h
    return canvas


# -------------------------------
# Main driver
# -------------------------------

def caption_frames_with_rerank(
    frames_csv: str | Path,
    out_jsonl: str | Path,
    cards_dir: Optional[str | Path] = None,
    device: str = "cpu",
    num_beams: int = 5,
    num_return: int = 5,
    max_new_tokens: int = 30,
    max_images: Optional[int] = None,
) -> Path:

    df = pd.read_csv(frames_csv)
    if max_images is not None:
        df = df.head(max_images)
    if df.empty:
        raise RuntimeError(f"No frames found in {frames_csv}")

    cap_pipe = load_blip(device=device)
    clip_model, clip_proc = load_clip()

    out_jsonl = Path(out_jsonl)
    ensure_dir(out_jsonl.parent)

    cards_dir_path = None
    if cards_dir is not None:
        cards_dir_path = ensure_dir(cards_dir)

    rows: List[Dict[str, Any]] = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Captioning (BLIP+CLIP)"):
        fp = r["frame_path"]
        ts = float(r.get("timestamp_sec", 0.0))
        frame_name = Path(fp).name

        try:
            best_caption = caption_one_image(
                fp, cap_pipe, clip_model, clip_proc,
                num_beams=num_beams, num_return=num_return, max_new_tokens=max_new_tokens
            )
        except Exception as e:
            logger.warning("Caption failed for %s: %s", fp, e)
            best_caption = ""

        rows.append({"frame_path": fp, "timestamp_sec": ts, "caption": best_caption})

        # Save a card image (optional)
        if cards_dir_path is not None:
            img = cv2.imread(fp)
            if img is not None:
                card = render_caption_card(img, best_caption, ts, frame_name)
                cv2.imwrite(str(cards_dir_path / f"cap_{Path(fp).stem}.jpg"), card)

    # Write JSONL
    with out_jsonl.open("w", encoding="utf-8") as f:
        for it in rows:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    logger.info("Wrote captions → %s (rows=%d)", out_jsonl, len(rows))
    if cards_dir_path:
        logger.info("Wrote caption cards → %s", cards_dir_path)
    return out_jsonl


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Task 1: BLIP captioning with CLIP re-ranking.")
    parser.add_argument("--frames_csv", required=True, help="Path to frames.csv from extraction step")
    parser.add_argument("--out_jsonl", required=True, help="Where to write captions.jsonl")
    parser.add_argument("--cards_dir", default=None, help="Optional directory to save caption cards (images)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--num_return", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    caption_frames_with_rerank(
        frames_csv=args.frames_csv,
        out_jsonl=args.out_jsonl,
        cards_dir=args.cards_dir,
        device=args.device,
        num_beams=args.num_beams,
        num_return=args.num_return,
        max_new_tokens=args.max_new_tokens,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
