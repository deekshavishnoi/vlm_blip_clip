"""
Task 1 — Summarize per-frame captions into a video-level description

WHAT THIS FILE DOES
-------------------
- Reads the per-frame captions JSONL produced by caption_frames.py.
- Orders them by timestamp.
- De-duplicates highly similar captions (simple, dependency-light heuristic).
- Produces a short, human-friendly paragraph describing the whole video.
- Writes the result to a plain text file (description.txt by default).

HEURISTIC
---------
- Normalize each caption (lowercase, strip punctuation/extra spaces).
- Keep captions in temporal order.
- Drop consecutive captions that are too similar (Jaccard token overlap ≥ 0.8).
- Keep a small set of representative captions (start / middle / end).
- Turn those into 2–5 short sentences.

OUTPUT
------
A single .txt file with a compact description.

USAGE (CLI)
-----------
python -m task1_video_captioning.src.summarize --cfg task1_video_captioning/configs/task1.yaml
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import re

from common.io_utils import read_yaml, load_jsonl, write_text, init_logger

logger = init_logger("task1.summarize")


# -----------------------
# Text helpers
# -----------------------

_PUNCT_RE = re.compile(r"[^\w\s]")

def _normalize_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse spaces."""
    s = s.lower().strip()
    s = _PUNCT_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _jaccard(a: str, b: str) -> float:
    """Very small similarity metric, token-based."""
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _dedupe_temporal(captions: List[Dict[str, Any]], thr: float = 0.8) -> List[Dict[str, Any]]:
    """
    Drop consecutive captions that are almost the same by Jaccard similarity.
    Keeps temporal order and returns a compressed list.
    """
    if not captions:
        return []
    cleaned = []
    prev_norm = None
    for item in captions:
        norm = _normalize_text(item["caption"])
        if prev_norm is None or _jaccard(prev_norm, norm) < thr:
            # Keep it
            cleaned.append({**item, "caption_norm": norm})
            prev_norm = norm
        # else: skip as near-duplicate
    return cleaned


def _pick_representatives(items: List[Dict[str, Any]], max_n: int = 5) -> List[Dict[str, Any]]:
    """
    Select up to max_n representative captions spread across the timeline.
    Simple uniform sampling over the compressed list.
    """
    if not items:
        return []
    if len(items) <= max_n:
        return items
    step = max(1, len(items) // max_n)
    reps = [items[i] for i in range(0, len(items), step)]
    return reps[:max_n]


def _to_paragraph(reps: List[Dict[str, Any]]) -> str:
    """
    Turn representative captions into a compact, readable paragraph.
    """
    if not reps:
        return "The video contains a short scene with limited visual content."

    # Ensure chronological order
    reps = sorted(reps, key=lambda x: x["timestamp_sec"])

    # Build simple narrative: beginning / middle / end
    parts = []
    if len(reps) == 1:
        parts.append(reps[0]["caption"])
    else:
        for i, r in enumerate(reps):
            c = r["caption"].strip()
            if i == 0:
                parts.append(f"The video opens with: {c}")
            elif i == len(reps) - 1:
                parts.append(f"Finally: {c}")
            else:
                parts.append(c)

    # Clean up whitespace and ensure ending period.
    paragraph = " ".join(parts)
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    if not paragraph.endswith((".", "!", "?")):
        paragraph += "."
    return paragraph


# -----------------------
# Public API
# -----------------------

def summarize_captions(
    captions_jsonl: str | Path,
    out_txt: str | Path,
    dedupe_threshold: float = 0.8,
    max_sentences: int = 5,
) -> str:
    """
    Summarize frame captions into a short description and write to disk.

    Args:
        captions_jsonl: Path to JSONL produced by caption_frames.py
        out_txt: Output path for the description text file
        dedupe_threshold: Jaccard similarity threshold for consecutive caption drop
        max_sentences: Max number of representative sentences in final summary

    Returns:
        The generated paragraph (string).
    """
    captions_jsonl = Path(captions_jsonl)
    out_txt = Path(out_txt)

    items = load_jsonl(captions_jsonl)
    if not items:
        logger.warning("No captions found in %s", captions_jsonl)
        paragraph = "No captions were generated for this video."
        write_text(out_txt, paragraph)
        return paragraph

    # Sort by time, then dedupe
    items = sorted(items, key=lambda x: x.get("timestamp_sec", 0.0))
    compressed = _dedupe_temporal(items, thr=dedupe_threshold)
    reps = _pick_representatives(compressed, max_n=max_sentences)
    paragraph = _to_paragraph(reps)

    write_text(out_txt, paragraph)
    logger.info("Wrote video description → %s", out_txt)
    return paragraph


# -----------------------
# CLI
# -----------------------

def main():

    import argparse
    parser = argparse.ArgumentParser(description="Summarize per-frame captions into a short video description.")
    parser.add_argument("--cfg", required=True, help="YAML config for Task 1.")
    parser.add_argument("--dedupe_thr", type=float, default=0.8, help="Jaccard threshold for near-duplicate removal.")
    parser.add_argument("--max_sentences", type=int, default=5, help="Max sentences in the final description.")
    args = parser.parse_args()

    cfg = read_yaml(args.cfg)
    summarize_captions(
        captions_jsonl=cfg["captions_path"],
        out_txt=cfg["video_description_path"],
        dedupe_threshold=args.dedupe_thr,
        max_sentences=args.max_sentences,
    )


if __name__ == "__main__":
    main()
