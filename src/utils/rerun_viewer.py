import time
import json
import yaml
import rerun as rr
import numpy as np
from PIL import Image
import urllib.parse
from scipy.spatial.transform import Rotation as R

# =========================================================
# Load dataset_config.yaml (camera intrinsics)
# =========================================================
CONFIG_PATH = "/disks/ssd1/kmw2622/workspace/semantic_forest_generation/config/dataset_config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

cam_cfg = config["dataset"]["camera"]

intr_resolution = (cam_cfg["width"], cam_cfg["height"])
intr_focal = (cam_cfg["fx"], cam_cfg["fy"])
intr_center = (cam_cfg["cx"], cam_cfg["cy"])


# =========================================================
# Initialize Rerun
# =========================================================
rr.init("coex_1f_scene_graph6", spawn=False)

server_uri = rr.serve_grpc()
rr.serve_web_viewer(connect_to=server_uri)

print("[RERUN VIEWER READY] URL:")
print("http://127.0.0.1:9090/?url=" + urllib.parse.quote(server_uri, safe=""))

# Set world coordinate convention
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)


# =========================================================
# Load processed graph (nodes + edges)
# =========================================================
with open(
    "/disks/ssd1/kmw2622/workspace/semantic_forest_generation/datasets/coex_1f_processed/topological_graph.json",
    "r"
) as f:
    data = json.load(f)

nodes = data["nodes"]
edges = data["edges"]

positions = np.array([n["position"] for n in nodes], dtype=np.float32)
colors = (np.random.rand(len(nodes), 3) * 255).astype(np.uint8)


# =========================================================
# Log static scene graph geometry (points + edges)
# =========================================================

# Points (all nodes)
rr.log(
    "world/points",
    rr.Points3D(
        positions,
        colors=colors,
        radii=0.15,
    ),
)

# Edges (topological graph)
edge_list = []
for e in edges:
    src, dst = e["src"], e["dst"]
    edge_list.append([positions[src], positions[dst]])

rr.log("world/edges", rr.LineStrips3D(edge_list))


# =========================================================
# Log Pinhole camera intrinsics (static)
# ---------------------------------------------------------
# This does NOT belong to any timeline.
# Only extrinsics + RGB belong to the timeline.
# =========================================================
rr.log(
    "world/cameras",
    rr.Pinhole(
        resolution=intr_resolution,
        focal_length=intr_focal,
        principal_point=intr_center,
        camera_xyz=rr.ViewCoordinates.RDF,  # OpenCV convention
    ),
)


# =========================================================
# Time-sequence logging (extrinsics + RGB + caption)
# =========================================================
for i, node in enumerate(nodes):

    # -----------------------------------------------------
    # Set timeline index
    # -----------------------------------------------------
    rr.set_time("image", sequence=i)

    # -----------------------------------------------------
    # RGB Image
    # -----------------------------------------------------
    try:
        img = Image.open(node["image"])
        rr.log("world/cameras/rgb", rr.Image(np.array(img)))
    except Exception as e:
        print(f"[Node {i}] Failed to load image:", e)

    # -----------------------------------------------------
    # Caption text
    # -----------------------------------------------------
    rr.log("world/cameras/caption", rr.TextDocument(node["caption"]))

    # -----------------------------------------------------
    # Camera EXTRINSICS (Transform3D)
    # -----------------------------------------------------
    pos = np.array(node["position"], dtype=np.float32)
    quat = np.array(node["quaternion"], dtype=np.float32)  # [x, y, z, w]
    rot_mat = R.from_quat([quat[0], quat[1], quat[2], quat[3]]).as_matrix().astype(np.float32)

    rr.log(
        "world/cameras",
        rr.Transform3D(
            translation=pos,
            mat3x3=rot_mat,
        ),
    )

print("✨ Rerun visualization running!")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
