# scripts/build_memory.py

import json
import yaml
import numpy as np
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from src.memory.builder import build_semantic_forest


CONFIG_PATH = "/disks/ssd1/kmw2622/workspace/embodied-rag/config/dataset_config.yaml"


# ---------------------------------------------------------
# Load config
# ---------------------------------------------------------
def load_config():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config["dataset"]


# ---------------------------------------------------------
# Load all node metadata (poses + captions + images)
# ---------------------------------------------------------
def load_graph_metadata(processed_root):
    graph_path = os.path.join(processed_root, "topological_graph.json")

    print("[LOAD] topological_graph.json →", graph_path)

    with open(graph_path, "r") as f:
        data = json.load(f)

    nodes = data["nodes"]

    positions = np.array([n["position"] for n in nodes], dtype=np.float32)
    captions = [n["caption"] for n in nodes]
    images = [n["image"] for n in nodes]
    quaternions = np.array([n["quaternion"] for n in nodes], dtype=np.float32)

    print(f"[DATA] Loaded {len(nodes)} nodes")
    print(" - positions:", positions.shape)
    print(" - captions:", len(captions))
    print(" - images:", len(images))
    print(" - quaternions:", quaternions.shape)

    return positions, captions, images, quaternions


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":

    # Load config
    cfg = load_config()
    processed_root = cfg["processed_root"]

    print("[CONFIG] processed_root =", processed_root)

    # Load graph metadata (pose + image + caption)
    positions, captions, images, quaternions = load_graph_metadata(processed_root)

    # Load embeddings
    emb_path = os.path.join(processed_root, "embeddings.npy")
    print("[LOAD] embeddings.npy →", emb_path)

    embeddings = np.load(emb_path)
    print(f"[DATA] Loaded embeddings: shape = {embeddings.shape}")

    # Build Semantic Forest
    print("[BUILD] Building Semantic Forest...")

    forest = build_semantic_forest(
        positions=positions,
        embeddings=embeddings,
        captions=captions,
        images=images,
        quaternions=quaternions,
        theta_spatial=10.0,
        alpha=0.3,
        cluster_threshold=0.4,
    )

    # Save result
    out_path = os.path.join(processed_root, "semantic_forest.json")

    with open(out_path, "w") as f:
        json.dump(forest, f, indent=2)

    print(f"[DONE] Saved semantic_forest.json → {out_path}")
