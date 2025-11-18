# scripts/04_build_graph.py

import os
import json
import yaml


def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "dataset_config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    processed_root = cfg["dataset"]["processed_root"]

    nodes_with_cap_path = os.path.join(processed_root, "nodes_with_captions.json")
    nodes_raw_path = os.path.join(processed_root, "nodes_raw.json")
    edges_path = os.path.join(processed_root, "edges.json")

    if os.path.exists(nodes_with_cap_path):
        nodes_path = nodes_with_cap_path
    else:
        nodes_path = nodes_raw_path

    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"노드 파일을 찾을 수 없습니다: {nodes_path}")
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"엣지 파일을 찾을 수 없습니다: {edges_path}")

    with open(nodes_path, "r") as f:
        nodes = json.load(f)

    with open(edges_path, "r") as f:
        edges_data = json.load(f)
        edges = edges_data.get("edges", [])
        edge_meta = edges_data.get("meta", {})

    graph = {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "dataset_name": cfg["dataset"]["name"],
            "alpha": edge_meta.get("alpha", None),
            "time_window": edge_meta.get("time_window", None),
            "num_nodes": len(nodes),
            "num_edges": len(edges)
        }
    }

    out_path = os.path.join(processed_root, "topological_graph.json")
    with open(out_path, "w") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    print(f"[INFO] topological_graph.json 저장 완료 → {out_path}")
    print(f"[INFO] 노드 수: {len(nodes)}, 엣지 수: {len(edges)}")


if __name__ == "__main__":
    main()
