# scripts/05_viz_graph.py

import os
import json
import yaml
import matplotlib.pyplot as plt
import numpy as np


def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "dataset_config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    cfg = load_config()
    processed_root = cfg["dataset"]["processed_root"]
    viz_dir = os.path.join(processed_root, "viz")
    ensure_dir(viz_dir)

    graph_path = os.path.join(processed_root, "topological_graph.json")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"topological_graph.json 이 없습니다: {graph_path}")

    with open(graph_path, "r") as f:
        graph = json.load(f)

    nodes = graph["nodes"]
    edges = graph["edges"]

    # node_id → index 매핑
    id_to_index = {n["node_id"]: i for i, n in enumerate(nodes)}

    xs = np.array([n["position"][0] for n in nodes], dtype=float)
    ys = np.array([n["position"][1] for n in nodes], dtype=float)

    plt.figure(figsize=(12, 10))

    # 우선 노드 그리기
    plt.scatter(xs, ys, s=10, c="black", alpha=0.7, label="nodes")

    # 엣지 타입별로 따로 그림
    for edge in edges:
        src_id = edge["src"]
        dst_id = edge["dst"]
        etype = edge.get("type", "sequence")

        if src_id not in id_to_index or dst_id not in id_to_index:
            continue

        i = id_to_index[src_id]
        j = id_to_index[dst_id]

        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[j], ys[j]

        if etype == "sequence":
            plt.plot([x1, x2], [y1, y2], color="blue", linewidth=0.6, alpha=0.5)
        else:
            # proximity 등 기타
            plt.plot([x1, x2], [y1, y2], color="red", linewidth=0.4, alpha=0.4)

    plt.title(f"Topological Graph – {graph['meta'].get('dataset_name', '')}")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(viz_dir, "graph_topological.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[INFO] 그래프 시각화 저장 완료 → {out_path}")


if __name__ == "__main__":
    main()
