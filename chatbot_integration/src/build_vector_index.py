"""
Task 3 — Build a vector index (FAISS) over per-frame descriptions

WHAT THIS FILE DOES
-------------------
- Loads the unified knowledge base (Parquet) built in build_knowledge_base.py
- Encodes each row's 'description' using a sentence embedding model
- L2-normalizes embeddings (so inner-product ≈ cosine similarity)
- Saves:
    - FAISS index file (embeddings.faiss)
    - Metadata JSON mapping index IDs → frame info (metadata.json)

"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

import json
import numpy as np
import pandas as pd

from common.io_utils import read_yaml, ensure_dir, init_logger

logger = init_logger("task3.index")

# Optional FAISS import with helpful error if missing
try:
    import faiss  # type: ignore
    _FAISS_OK = True
except Exception as e:
    faiss = None  # type: ignore
    _FAISS_OK = False
    _FAISS_ERR = e


def _load_kb(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Knowledge base not found: {p}")
    df = pd.read_parquet(p)
    if "description" not in df.columns:
        raise KeyError("knowledge_base.parquet is missing the 'description' column.")
    if "frame_path" not in df.columns or "timestamp_sec" not in df.columns:
        raise KeyError("knowledge_base.parquet must contain 'frame_path' and 'timestamp_sec'.")
    # Fill any NAs to keep encoders happy
    df["description"] = df["description"].fillna("No description.")
    return df


def _load_encoder(model_name: str, device: str = "cpu"):
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: %s (device=%s)", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    return model


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    """
    L2-normalize embeddings row-wise so that inner product equals cosine similarity.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def _build_faiss_index(emb: np.ndarray):
    """
    Create a FAISS inner-product index and add normalized embeddings.
    """
    if not _FAISS_OK:
        raise RuntimeError(
            f"faiss-cpu is not installed or failed to import: {_FAISS_ERR}\n"
            "Install with: pip install faiss-cpu"
        )
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product (works as cosine if vectors are normalized)
    index.add(emb.astype("float32"))
    return index


def build_vector_index(
    kb_path: str | Path,
    index_path: str | Path,
    metadata_path: str | Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
) -> None:
    """
    Build FAISS index + metadata from the knowledge base.
    """
    # 1) Load KB
    df = _load_kb(kb_path)
    logger.info("Loaded knowledge base: %d rows", len(df))

    # 2) Encode descriptions
    encoder = _load_encoder(model_name, device=device)
    texts: List[str] = df["description"].astype(str).tolist()

    # Encode in small batches to keep memory low
    batch = 256
    vecs: List[np.ndarray] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        v = encoder.encode(chunk, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)
        vecs.append(v)
    emb = np.vstack(vecs)
    emb = _normalize_rows(emb)  # cosine-ready
    logger.info("Encoded descriptions → embeddings shape: %s", emb.shape)

    # 3) Build FAISS index
    index = _build_faiss_index(emb)

    # 4) Save index and metadata
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)
    ensure_dir(index_path.parent)
    ensure_dir(metadata_path.parent)

    faiss.write_index(index, str(index_path))
    logger.info("Wrote FAISS index → %s", index_path)

    # Store minimal per-vector metadata (same row order as KB)
    # Keep it light: avoid dumping very long strings repeatedly
    meta_rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        meta_rows.append(
            {
                "frame_path": str(r["frame_path"]),
                "timestamp_sec": float(r["timestamp_sec"]),
                "caption": str(r.get("caption", ""))[:300],
                "scene_label": str(r.get("scene_label", "")),
                "objects_top": list(r.get("objects_top", [])) if isinstance(r.get("objects_top", []), list) else [],
            }
        )

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump({"rows": meta_rows}, f, ensure_ascii=False)

    logger.info("Wrote metadata → %s (rows=%d)", metadata_path, len(meta_rows))
    logger.info("Vector index build complete.")


# -----------------------
# CLI
# -----------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build FAISS index over frame descriptions.")
    parser.add_argument("--cfg", required=True, help="Path to task3.yaml")
    args = parser.parse_args()

    cfg = read_yaml(args.cfg)
    build_vector_index(
        kb_path=cfg["knowledge_base_path"],
        index_path=cfg["faiss_index_path"],
        metadata_path=cfg["metadata_path"],
        model_name=cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        device=cfg.get("device", "cpu"),
    )


if __name__ == "__main__":
    main()
