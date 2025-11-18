# scripts/03_build_edges.py

import os
import json
import yaml
import argparse
import math


def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "dataset_config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def euclidean_dist3(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=3.0,
                        help="proximity edge 거리 임계값 (미터)")
    parser.add_argument("--time_window", type=int, default=None,
                        help="proximity edge를 만들 때 시간적으로 이 정도 인덱스 차이까지만 고려 (O(N^2) 방지용)")
    args = parser.parse_args()

    cfg = load_config()
    processed_root = cfg["dataset"]["processed_root"]

    nodes_with_cap_path = os.path.join(processed_root, "nodes_with_captions.json")
    nodes_raw_path = os.path.join(processed_root, "nodes_raw.json")

    if os.path.exists(nodes_with_cap_path):
        nodes_path = nodes_with_cap_path
    else:
        nodes_path = nodes_raw_path

    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"노드 파일을 찾을 수 없습니다: {nodes_path}")

    with open(nodes_path, "r") as f:
        nodes = json.load(f)

    if len(nodes) < 2:
        print("[INFO] 노드 개수가 2개 미만입니다. 엣지를 만들지 않습니다.")
        return

    # node_id → index 매핑 (혹시 node_id가 0..N-1이 아닐 수도 있으므로)
    id_to_index = {n["node_id"]: i for i, n in enumerate(nodes)}

    # 시퀀스는 nodes 배열 순서 그대로 사용 (이미 timestamp 순으로 만들어져 있다고 가정)
    edges = []

    # sequence edges
    for i in range(len(nodes) - 1):
        src = nodes[i]["node_id"]
        dst = nodes[i + 1]["node_id"]
        edges.append({"src": src, "dst": dst, "type": "sequence"})

    # proximity edges
    N = len(nodes)
    window = args.time_window if args.time_window is not None else N

    for i in range(N):
        pos_i = nodes[i]["position"]
        for j in range(i + 1, min(N, i + 1 + window)):
            pos_j = nodes[j]["position"]
            d = euclidean_dist3(pos_i, pos_j)
            if d < args.alpha:
                src = nodes[i]["node_id"]
                dst = nodes[j]["node_id"]
                edges.append({"src": src, "dst": dst, "type": "proximity"})

    edges_out_path = os.path.join(processed_root, "edges.json")
    out = {
        "edges": edges,
        "meta": {
            "alpha": args.alpha,
            "time_window": args.time_window
        }
    }

    with open(edges_out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[INFO] edges.json 저장 완료 → {edges_out_path}")
    print(f"[INFO] 총 edge 수: {len(edges)} (sequence + proximity)")


if __name__ == "__main__":
    main()
