"""
Task 1 — Frame extraction (OpenCV)

WHAT THIS FILE DOES
-------------------
- Opens the local MP4 video.
- Samples frames at a fixed rate (e.g., 1 frame per two seconds).
- Saves frames as images to disk.
- Returns a manifest (pandas DataFrame) with:
  [frame_path, timestamp_sec, fps, width, height]

USAGE (called from run_task1.py)
--------------------------------
df = extract_frames(
    video_path="task1_video_captioning/data/raw/your_video.mp4",
    output_dir="task1_video_captioning/data/interim/frames",
    frames_per_second=1.0,
    image_ext=".jpg",
    max_frames=None,
)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
from tqdm import tqdm

from common.io_utils import ensure_dir
from common.io_utils import init_logger


logger = init_logger("task1.extract")


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    frames_per_second: float = 0.5,
    image_ext: str = ".jpg",
    max_frames: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extract frames at a fixed sampling rate and return a DataFrame.

    Args:
        video_path: Path to the input video file (.mp4).
        output_dir: Directory where extracted frame images will be written.
        frames_per_second: Target sampling rate (e.g., 0.5 → one frame per two seconds).
        image_ext: Image extension for saved frames (".jpg" or ".png").
        max_frames: Optional hard cap on the number of frames to save.

    Returns:
        DataFrame with columns:
            - frame_path: str (absolute or project-relative path to the saved image)
            - timestamp_sec: float (when the frame appears in video)
            - fps: float (native FPS reported by the video)
            - width: int
            - height: int
    """
    video_path = Path(video_path)
    output_dir = ensure_dir(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / native_fps if native_fps > 0 else 0.0

    logger.info(
        "Opened video: %s | native_fps=%.3f | frames=%d | duration=%.2fs | size=%dx%d",
        video_path.name, native_fps, total_frames, duration_sec, width, height
    )

    # Decide how often we save a frame, in terms of raw frame indices.
    # Example: if native_fps=29.97 and frames_per_second=1.0 → step ≈ 30
    step = max(1, int(round(native_fps / frames_per_second))) if frames_per_second > 0 else 1

    rows = []
    saved = 0
    frame_idx = 0

    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while True:
            # grab() is faster than read(); we'll retrieve only when needed
            ret = cap.grab()
            if not ret:
                break

            if frame_idx % step == 0:
                # Actually decode the grabbed frame
                ok, frame = cap.retrieve()
                if not ok:
                    break

                ts = frame_idx / native_fps if native_fps > 0 else 0.0
                fname = f"frame_{saved:06d}{image_ext}"
                out_path = output_dir / fname

                # OpenCV uses BGR; writing directly is fine
                write_ok = cv2.imwrite(str(out_path), frame)
                if not write_ok:
                    raise RuntimeError(f"Failed to write frame: {out_path}")

                rows.append(
                    {
                        "frame_path": str(out_path),
                        "timestamp_sec": float(ts),
                        "fps": float(native_fps),
                        "width": width,
                        "height": height,
                    }
                )
                saved += 1

                if max_frames is not None and saved >= max_frames:
                    break

            frame_idx += 1
            pbar.update(1)

    cap.release()

    df = pd.DataFrame(rows)
    logger.info("Saved %d frames → %s", len(df), output_dir)
    return df
