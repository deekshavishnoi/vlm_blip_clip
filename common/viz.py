"""
common/viz.py

Tiny visualization helpers for quick sanity checks and reporting.

WHY THIS FILE EXISTS
--------------------
You’ll want a few simple plots and overlays to *show* what your pipeline is doing:
- Save a grid of sample frames for the README
- Draw detection boxes on a frame
- Plot a bar chart of top scenes/objects

These functions are intentionally minimal (matplotlib + PIL only) to keep
dependencies light and CPU-friendly.
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence, Optional, List, Dict, Any

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


# -----------------------
# Frame grids for README
# -----------------------

def save_image_grid(
    image_paths: Sequence[str | Path],
    out_path: str | Path,
    cols: int = 3,
    max_images: int = 9,
    figsize: tuple[int, int] = (10, 8),
    titles: Optional[Sequence[str]] = None,
) -> None:
    """
    Save a simple grid of images (no external styling).
    Use in Task 1/2 READMEs to show representative frames.
    """
    paths = list(image_paths)[:max_images]
    n = len(paths)
    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=figsize)
    for i, p in enumerate(paths, start=1):
        ax = plt.subplot(rows, cols, i)
        ax.imshow(Image.open(p))
        ax.axis("off")
        if titles and i - 1 < len(titles):
            ax.set_title(titles[i - 1], fontsize=10)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------
# Draw boxes & captions
# -----------------------

def draw_boxes_on_image(
    image_path: str | Path,
    boxes: Sequence[Dict[str, Any]],
    out_path: str | Path,
    labels_key: str = "label",
    xyxy_norm: bool = True,
) -> None:
    """
    Draw detection boxes onto an image and save it.

    Args:
        image_path: source frame.
        boxes: sequence of dicts. Each dict should have:
               - x1, y1, x2, y2: floats (normalized [0,1] if xyxy_norm=True)
               - label: str (optional but recommended)
               - confidence: float (optional)
        out_path: where to save the annotated image.
        labels_key: the key name for the class label in dicts.
        xyxy_norm: set True if coordinates are normalized to [0,1].

    This is used in Task 2 to visually verify YOLO detections.
    """
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)

    for b in boxes:
        x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
        if xyxy_norm:
            x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
        # rectangle + optional text
        draw.rectangle([(x1, y1), (x2, y2)], outline="white", width=2)
        label = str(b.get(labels_key, ""))
        conf = b.get("confidence", None)
        if label or conf is not None:
            text = f"{label}" if conf is None else f"{label} {conf:.2f}"
            # simple text box
            tw, th = draw.textlength(text), 12
            draw.rectangle([(x1, max(0, y1 - th - 4)), (x1 + tw + 6, y1)], fill="white")
            draw.text((x1 + 3, y1 - th - 2), text, fill="black")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def overlay_caption(
    image_path: str | Path,
    caption: str,
    out_path: str | Path,
) -> None:
    """
    Write a short caption onto the bottom of an image and save it.
    Handy for Task 1 to present (frame, caption) pairs in the README.
    """
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)

    # simple bottom strip
    pad = 8
    strip_h = int(min(60, 0.2 * h))
    draw.rectangle([(0, h - strip_h), (w, h)], fill="white")

    # wrap text roughly to fit width
    max_chars = max(10, int(w / 10))
    words = caption.split()
    lines: List[str] = []
    line = []
    for w_ in words:
        if len(" ".join(line + [w_])) <= max_chars:
            line.append(w_)
        else:
            lines.append(" ".join(line))
            line = [w_]
    if line:
        lines.append(" ".join(line))

    y = h - strip_h + pad
    for ln in lines[:3]:  # keep it compact
        draw.text((pad, y), ln, fill="black")
        y += 14

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


# -----------------------
# Simple bar chart (scenes/objects)
# -----------------------

def save_bar_chart(labels: Sequence[str], counts: Sequence[int], title: str, out_path: str | Path) -> None:
    """
    Minimal bar chart for README (no styling).
    Used to show top-N objects or scene distribution in Task 2.
    """
    fig = plt.figure(figsize=(8, 4))
    plt.bar(range(len(labels)), counts)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.title(title)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
