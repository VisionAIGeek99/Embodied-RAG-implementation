# scripts/embed_nodes.py

import json
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer
import os

CONFIG_PATH = "/disks/ssd1/kmw2622/workspace/semantic_forest_generation/config/dataset_config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config["dataset"]


def compute_embeddings():
    # ---------------------------------------------------------
    # Load config
    # ---------------------------------------------------------
    cfg = load_config()
    processed_root = cfg["processed_root"]

    graph_path = os.path.join(processed_root, "topological_graph.json")
    emb_path = os.path.join(processed_root, "embeddings.npy")

    print("[CONFIG] processed_root:", processed_root)
    print("[LOAD] topological_graph.json →", graph_path)

    # ---------------------------------------------------------
    # Load nodes
    # ---------------------------------------------------------
    with open(graph_path, "r") as f:
        data = json.load(f)

    nodes = data["nodes"]
    captions = [n["caption"] for n in nodes]

    print(f"[DATA] Loaded {len(captions)} captions.")

    # ---------------------------------------------------------
    # Load embedding model
    # ---------------------------------------------------------
    print("[MODEL] Loading BGE-large-en-v1.5...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    # ---------------------------------------------------------
    # Encode captions
    # ---------------------------------------------------------
    print("[EMBED] Encoding captions...")
    embeddings = model.encode(
        captions,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=False,   # cosine similarity에서 정규화 처리
    )

    # ---------------------------------------------------------
    # Save result
    # ---------------------------------------------------------
    np.save(emb_path, embeddings)
    print(f"[DONE] Saved embeddings.npy → {emb_path}")


if __name__ == "__main__":
    compute_embeddings()
