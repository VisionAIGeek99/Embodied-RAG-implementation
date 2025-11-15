import os
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import quaternion
from kapture.io.csv import kapture_from_dir

def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "dataset_config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    cfg = load_config()
    RAW_ROOT = cfg["dataset"]["raw_root"]
    PROCESSED = cfg["dataset"]["processed_root"]
    TARGET_CAM = cfg["dataset"]["target_camera"]
    MAX_NODES = cfg["dataset"]["max_nodes"]

    FRAME_DIR = os.path.join(PROCESSED, "frames")
    VIZ_DIR = os.path.join(PROCESSED, "viz")
    ensure_dir(FRAME_DIR)
    ensure_dir(VIZ_DIR)

    # --------------------------------------
    # Load Kapture dataset
    # --------------------------------------
    print("[INFO] Loading kapture...")
    kdata = kapture_from_dir(RAW_ROOT)
    trajectories = kdata.trajectories
    records_camera = kdata.records_camera

    xs, ys, zs = [], [], []
    nodes = []
    timestamps_sorted = sorted(trajectories.keys())

    # --------------------------------------
    # Extract nodes
    # --------------------------------------
    node_id = 0

    for timestamp in timestamps_sorted:
        if node_id >= MAX_NODES:
            break

        if TARGET_CAM not in trajectories[timestamp]:
            continue

        pose = trajectories[timestamp][TARGET_CAM]

        # pose: world → camera
        # camera → world = pose.inverse()
        world_pose = pose.inverse()
        t = world_pose.t.flatten()
        rx, ry, rz = float(t[0]), float(t[1]), float(t[2])

        xs.append(rx)
        ys.append(ry)
        zs.append(rz)

        # 이미지 경로
        rel_path = None
        if timestamp in records_camera and TARGET_CAM in records_camera[timestamp]:
            rel_path = records_camera[timestamp][TARGET_CAM]
            img_path = os.path.join(RAW_ROOT, "sensors", "records_data", rel_path)
        else:
            img_path = None

        # 이미지 저장
        saved_img = None
        if img_path and os.path.exists(img_path):
            img = cv2.imread(img_path)
            save_path = os.path.join(FRAME_DIR, f"{node_id:05d}.jpg")
            cv2.imwrite(save_path, img)
            saved_img = save_path

        nodes.append({
            "node_id": node_id,
            "timestamp": int(timestamp),
            "position": [rx, ry, rz],
            "image": saved_img
        })

        node_id += 1

    # --------------------------------------
    # Save raw nodes JSON
    # --------------------------------------
    nodes_json = os.path.join(PROCESSED, "nodes_raw.json")
    with open(nodes_json, "w") as f:
        json.dump(nodes, f, indent=2)

    print(f"[INFO] Saved nodes_raw.json → {nodes_json}")

    # --------------------------------------
    # Visualization (XY only)
    # --------------------------------------
    xs = np.array(xs)
    ys = np.array(ys)

    plt.figure(figsize=(12, 10))
    colors = np.linspace(0, 1, len(xs))
    plt.scatter(xs, ys, c=colors, cmap='jet', s=10)
    plt.plot(xs, ys, color="gray", linewidth=0.7)
    plt.title("Trajectory (camera positions)")
    plt.axis("equal")
    plt.grid(True)

    viz_path = os.path.join(VIZ_DIR, "trajectory_xy.png")
    plt.savefig(viz_path, dpi=300)
    plt.close()

    print(f"[INFO] Saved trajectory → {viz_path}")


if __name__ == "__main__":
    main()
