"""
Task 3 — Query Interface (supports SBERT/FAISS and TF-IDF backends)

WHAT THIS FILE DOES
-------------------
- Loads the index + metadata built in Task 3.
- Accepts a natural-language query.
- Returns top-k matching frames with timestamps (+ optional preview image list).

"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import numpy as np

from common.io_utils import read_yaml, init_logger

logger = init_logger("task3.query")

# Optional imports guarded for cleaner errors
try:
    import faiss  # type: ignore
    _FAISS_OK = True
except Exception:
    faiss = None  # type: ignore
    _FAISS_OK = False


# -----------------------
# SBERT backend
# -----------------------

def _load_sbert(cfg: Dict[str, Any]):
    from sentence_transformers import SentenceTransformer
    index_path = Path(cfg["faiss_index_path"])
    meta_path = Path(cfg["metadata_path"])
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    logger.info("Loading FAISS index: %s", index_path)
    index = faiss.read_index(str(index_path))
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)["rows"]

    model_name = cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    device = cfg.get("device", "cpu")
    logger.info("Loading SentenceTransformer: %s (device=%s)", model_name, device)
    enc = SentenceTransformer(model_name, device=device)
    return enc, index, meta


def _sbert_search(enc, index, meta: List[Dict[str, Any]], query: str, k: int = 5):
    qv = enc.encode([query], convert_to_numpy=True, normalize_embeddings=False)
    # cosine-ready normalize
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)
    D, I = index.search(qv.astype("float32"), k)
    scores = D[0].tolist()
    idxs = I[0].tolist()
    results = []
    for score, i in zip(scores, idxs):
        if i < 0 or i >= len(meta):
            continue
        m = meta[i]
        results.append({
            "rank": len(results) + 1,
            "score": float(score),
            "frame_path": m["frame_path"],
            "timestamp_sec": float(m["timestamp_sec"]),
            "scene_label": m.get("scene_label", ""),
            "objects_top": m.get("objects_top", []),
            "caption": m.get("caption", ""),
        })
    return results


# -----------------------
# TF-IDF backend (fallback)
# -----------------------

def _load_tfidf(cfg: Dict[str, Any]):
    from scipy.sparse import load_npz
    import joblib
    mat_path = Path(cfg["tfidf_matrix_path"])
    vec_path = Path(cfg["tfidf_model_path"])
    meta_path = Path(cfg["metadata_path"])
    if not mat_path.exists() or not vec_path.exists():
        raise FileNotFoundError("TF-IDF files not found. Build index with TF-IDF backend.")
    X = load_npz(mat_path)
    vect = joblib.load(vec_path)
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)["rows"]
    return vect, X, meta


def _tfidf_search(vect, X, meta: List[Dict[str, Any]], query: str, k: int = 5):
    from sklearn.metrics.pairwise import cosine_similarity
    q = vect.transform([query])
    sims = cosine_similarity(q, X).ravel()
    top_idx = np.argsort(-sims)[:k]
    results = []
    for rnk, i in enumerate(top_idx, start=1):
        m = meta[int(i)]
        results.append({
            "rank": rnk,
            "score": float(sims[i]),
            "frame_path": m["frame_path"],
            "timestamp_sec": float(m["timestamp_sec"]),
            "scene_label": m.get("scene_label", ""),
            "objects_top": m.get("objects_top", []),
            "caption": m.get("caption", ""),
        })
    return results


# -----------------------
# Public entry-point
# -----------------------

def query_frames(cfg_path: str | Path, query: str, k: int = 5) -> List[Dict[str, Any]]:
    cfg = read_yaml(cfg_path)
    backend = cfg.get("embedding_backend", "sbert").lower()

    if backend == "sbert":
        if not _FAISS_OK:
            raise RuntimeError("FAISS not available; install faiss-cpu or switch backend to 'tfidf'.")
        enc, index, meta = _load_sbert(cfg)
        results = _sbert_search(enc, index, meta, query=query, k=k)
    else:
        vect, X, meta = _load_tfidf(cfg)
        results = _tfidf_search(vect, X, meta, query=query, k=k)

    return results


def _print_results(rows: List[Dict[str, Any]]):
    if not rows:
        print("No results.")
        return
    print("\nTop matches:")
    print("-" * 80)
    for r in rows:
        t = f"{r['timestamp_sec']:7.2f}s"
        objs = ", ".join(r.get("objects_top", [])[:4])
        print(f"[{r['rank']:>2}] score={r['score']:.3f}  time={t}  scene={r['scene_label'] or '-'}")
        print(f"     frame: {r['frame_path']}")
        if objs:
            print(f"     objs : {objs}")
        if r.get("caption"):
            print(f"     cap  : {r['caption'][:120]}")
        print("-" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Query video knowledge base (Task 3).")
    parser.add_argument("--cfg", required=True, help="Path to task3.yaml")
    parser.add_argument("--query", required=True, help="Natural-language search query")
    parser.add_argument("--k", type=int, default=5, help="Top-k frames to return")
    args = parser.parse_args()

    rows = query_frames(args.cfg, query=args.query, k=args.k)
    _print_results(rows)


if __name__ == "__main__":
    main()
