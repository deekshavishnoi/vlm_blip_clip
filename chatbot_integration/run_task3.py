"""
Task 3 — End-to-end runner (CLI)

Provides:
  - kb       : build knowledge_base.parquet
  - index    : build vector index (SBERT/FAISS by default; TF-IDF fallback supported)
  - query    : run a natural-language query over the index
  - all      : kb -> index

Usage:
  python -m task3_chatbot_integration.run_task3 all  --cfg task3_chatbot_integration/configs/task3.yaml
  python -m task3_chatbot_integration.run_task3 kb   --cfg ...
  python -m task3_chatbot_integration.run_task3 index --cfg ...
  python -m task3_chatbot_integration.run_task3 query --cfg ... --q "cars at a crosswalk" --k 5
"""

from __future__ import annotations
import argparse

from common.io_utils import read_yaml, init_logger
from task3_chatbot_integration.src.build_knowledge_base import build_knowledge_base
from task3_chatbot_integration.src.build_vector_index import build_vector_index
from task3_chatbot_integration.src.query_interface import query_frames

logger = init_logger("task3.runner")


def main():
    parser = argparse.ArgumentParser(description="Task 3 runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_shared(p):
        p.add_argument("--cfg", required=True, help="Path to YAML config")

    p_kb = sub.add_parser("kb", help="Build knowledge base")
    add_shared(p_kb)

    p_idx = sub.add_parser("index", help="Build vector index")
    add_shared(p_idx)

    p_q = sub.add_parser("query", help="Query the index")
    add_shared(p_q)
    p_q.add_argument("--q", required=True, help="Natural-language query")
    p_q.add_argument("--k", type=int, default=5)

    p_all = sub.add_parser("all", help="Build KB then index")
    add_shared(p_all)

    args = parser.parse_args()
    cfg = read_yaml(args.cfg)

    if args.cmd == "kb":
        build_knowledge_base(
            captions_path=cfg["captions_path"],
            objects_path=cfg["objects_path"],
            scenes_path=cfg["scenes_path"],
            out_parquet=cfg["knowledge_base_path"],
            min_obj_conf=0.25,
            top_n_objects=5,
        )

    elif args.cmd == "index":
        build_vector_index(
            kb_path=cfg["knowledge_base_path"],
            index_path=cfg["faiss_index_path"],
            metadata_path=cfg["metadata_path"],
            model_name=cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            device=cfg.get("device", "cpu"),
        )

    elif args.cmd == "query":
        rows = query_frames(args.cfg, query=args.q, k=args.k)
        # pretty print is inside query_interface.main, but we print minimal ack here
        logger.info("Returned %d results.", len(rows))
        # Print to console too
        from task3_chatbot_integration.src.query_interface import _print_results
        _print_results(rows)

    elif args.cmd == "all":
        # kb -> index
        build_knowledge_base(
            captions_path=cfg["captions_path"],
            objects_path=cfg["objects_path"],
            scenes_path=cfg["scenes_path"],
            out_parquet=cfg["knowledge_base_path"],
            min_obj_conf=0.25,
            top_n_objects=5,
        )
        build_vector_index(
            kb_path=cfg["knowledge_base_path"],
            index_path=cfg["faiss_index_path"],
            metadata_path=cfg["metadata_path"],
            model_name=cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            device=cfg.get("device", "cpu"),
           )

    logger.info("Task 3 done.")


if __name__ == "__main__":
    main()
