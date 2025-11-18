# src/utils/rerun_viewer.py

import os
import json
import time
import yaml
import urllib.parse

import numpy as np
import rerun as rr
from PIL import Image
from scipy.spatial.transform import Rotation as R


# =========================================================
# Paths / Config
# =========================================================
CONFIG_PATH = "/disks/ssd1/kmw2622/workspace/embodied-rag/config/dataset_config.yaml"

Z_STEP = 10.0          # level 당 Z offset
AREA_POINT_RADIUS = 0.4
LEAF_POINT_RADIUS = 0.2


def load_config():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["dataset"]


def main():
    cfg = load_config()
    processed_root = cfg["processed_root"]

    # -----------------------------------------------------
    # Load semantic forest
    # -----------------------------------------------------
    forest_path = os.path.join(processed_root, "semantic_forest.json")
    print("[LOAD] semantic_forest.json →", forest_path)

    with open(forest_path, "r") as f:
        forest = json.load(f)

    nodes = forest["nodes"]         # dict: id -> node dict
    root_id = forest["root"]
    print("[FOREST] root =", root_id)
    print("[FOREST] #nodes =", len(nodes))

    # -----------------------------------------------------
    # Load camera intrinsics
    # -----------------------------------------------------
    cam_cfg = cfg["camera"]
    intr_resolution = (cam_cfg["width"], cam_cfg["height"])
    intr_focal = (cam_cfg["fx"], cam_cfg["fy"])
    intr_center = (cam_cfg["cx"], cam_cfg["cy"])

    # =====================================================
    # Init Rerun
    # =====================================================
    rr.init("semantic_forest_viewer", spawn=False)

    server_uri = rr.serve_grpc()
    rr.serve_web_viewer(connect_to=server_uri)

    print("[RERUN VIEWER READY] URL:")
    print("http://127.0.0.1:9090/?url=" + urllib.parse.quote(server_uri, safe=""))

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)

    # Static Pinhole camera model (intrinsics only)
    rr.log(
        "world/cameras",
        rr.Pinhole(
            resolution=intr_resolution,
            focal_length=intr_focal,
            principal_point=intr_center,
            camera_xyz=rr.ViewCoordinates.RDF,  # OpenCV-style
        ),
    )

    # =====================================================
    # Compute 3D position for visualization (Z-level)
    # =====================================================
    def viz_position(node_dict):
        base_pos = np.array(node_dict["position"], dtype=np.float32)
        level = int(node_dict["level"])
        pos = base_pos.copy()
        pos[2] += level * Z_STEP
        return pos

    node_viz_pos = {nid: viz_position(nd) for nid, nd in nodes.items()}

    # Leaf / area 분리
    leaf_ids = [nid for nid, nd in nodes.items() if nd["type"] == "leaf"]
    area_ids = [nid for nid, nd in nodes.items() if nd["type"] != "leaf"]

    # -----------------------------------------------------
    # Points for leaf nodes
    # -----------------------------------------------------
    if leaf_ids:
        leaf_pos = np.array([node_viz_pos[nid] for nid in leaf_ids], dtype=np.float32)
        leaf_colors = np.array(
            [np.array([80, 160, 255], dtype=np.uint8) for _ in leaf_ids],
            dtype=np.uint8,
        )
        rr.log(
            "world/forest/leaf_points",
            rr.Points3D(
                leaf_pos,
                colors=leaf_colors,
                radii=LEAF_POINT_RADIUS,
            ),
        )

    # -----------------------------------------------------
    # Points for area nodes
    # -----------------------------------------------------
    if area_ids:
        area_pos = np.array([node_viz_pos[nid] for nid in area_ids], dtype=np.float32)
        area_colors = np.array(
            [np.array([255, 180, 0], dtype=np.uint8) for _ in area_ids],
            dtype=np.uint8,
        )
        rr.log(
            "world/forest/area_points",
            rr.Points3D(
                area_pos,
                colors=area_colors,
                radii=AREA_POINT_RADIUS,
                labels=area_ids, 
                show_labels=True
            ),
        )

    # -----------------------------------------------------
    # Hierarchy edges (parent → child)
    # -----------------------------------------------------
    edge_list = []
    for nid, nd in nodes.items():
        for child_id in nd["children"]:
            if child_id not in node_viz_pos:
                continue
            p_parent = node_viz_pos[nid]
            p_child = node_viz_pos[child_id]
            edge_list.append([p_parent, p_child])

    if edge_list:
        rr.log("world/forest/edges", rr.LineStrips3D(edge_list))

    # =====================================================
    # 추가: viewpoint(topological_graph) edges
    #   - L0_i 노드들끼리 원래 topological_graph.json의 edge를 다시 그림
    # =====================================================
    topo_path = os.path.join(processed_root, "topological_graph.json")
    print("[LOAD] topological_graph.json →", topo_path)

    try:
        with open(topo_path, "r") as f:
            topo = json.load(f)

        topo_edges = topo["edges"]

        vp_edge_list = []

        # topological_graph의 src/dst 인덱스를 L0_i로 매핑
        for e in topo_edges:
            src_idx = e["src"]
            dst_idx = e["dst"]

            src_id = f"L0_{src_idx}"
            dst_id = f"L0_{dst_idx}"

            if src_id not in node_viz_pos or dst_id not in node_viz_pos:
                continue

            p1 = node_viz_pos[src_id]   # level 0 이라 z-offset 없음
            p2 = node_viz_pos[dst_id]
            vp_edge_list.append([p1, p2])

        if vp_edge_list:
            rr.log(
                "world/forest/viewpoint_edges",
                rr.LineStrips3D(
                    vp_edge_list,
                    colors=[[0, 150, 255]],  # 파란색
                ),
            )
            print(f"[VIEWPOINT] logged {len(vp_edge_list)} edges from topological_graph")

    except Exception as e:
        print("[WARN] failed to load viewpoint edges from topological_graph:", e)

    # =====================================================
    # Timeline logging
    #   1) leaf: 이미지 + 캡션 + 실제 pose
    #   2) area: 요약 캡션만, level 순서대로
    # =====================================================

    # leaf: id 기준 정렬
    leaf_ids_sorted = sorted(leaf_ids, key=lambda x: int(x.split("_")[1]))

    for t, nid in enumerate(leaf_ids_sorted):
        nd = nodes[nid]
        rr.set_time("image", sequence=t)

        # 이미지
        img_path = nd.get("image")
        if img_path:
            try:
                img = Image.open(img_path)
                rr.log("world/cameras/rgb", rr.Image(np.array(img)))
            except Exception as e:
                print(f"[WARN] no image for {nid} ({img_path}):", e)
        else:
            print(f"[WARN] no image field for {nid}")

        # 캡션: raw_caption 있으면 그거, 없으면 summary
        caption_text = nd.get("raw_caption") or nd.get("summary", "")
        rr.log("world/cameras/caption", rr.TextDocument(caption_text))

        # 카메라 pose: 실제 position (z-offset 없이) + quaternion
        try:
            pos = np.array(nd["position"], dtype=np.float32)
            quat = np.array(nd["quaternion"], dtype=np.float32)  # [x,y,z,w]
            rot_mat = R.from_quat(quat).as_matrix().astype(np.float32)

            rr.log(
                "world/cameras",
                rr.Transform3D(
                    translation=pos,
                    mat3x3=rot_mat,
                ),
            )
        except KeyError as e:
            print(f"[WARN] missing pose field for {nid}:", e)

    # -----------------------------------------------------
    # area 노드들: summary를 나중 타임스텝에 순서대로 기록
    #   level 오름차순 → id 오름차순
    # -----------------------------------------------------
    base_t = len(leaf_ids_sorted)
    area_sorted = sorted(
        area_ids,
        key=lambda x: (int(nodes[x]["level"]), x),
    )

    for dt, nid in enumerate(area_sorted):
        nd = nodes[nid]
        rr.set_time("image", sequence=base_t + dt)

        # position (viz용 z-offset 포함 transform)
        pos = node_viz_pos[nid]
        rr.log(
            f"world/areas/{nid}",
            rr.Transform3D(translation=pos),
        )

        rr.log(
            f"world/areas/{nid}/summary",
            rr.TextDocument(nd.get("summary", "")),
        )

    print("✨ Semantic forest visualization running!")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
