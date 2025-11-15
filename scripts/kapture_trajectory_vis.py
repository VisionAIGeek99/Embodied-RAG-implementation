import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import quaternion
from kapture.io.csv import kapture_from_dir

# -------------------------
# 1. Load Kapture dataset
# -------------------------
root = "/disks/ssd1/kmw2622/dataset/coex_1F_release_mapping/1F/release/mapping"
kdata = kapture_from_dir(root)

trajectories = kdata.trajectories
records_camera = kdata.records_camera

TARGET_CAM = "40027089_00"

xs, ys, zs = [], [], []
node_ids = []
image_paths = []
valid_timestamps = []

count = 0
node_id = 0

save_img_dir = "saved_frames"
os.makedirs(save_img_dir, exist_ok=True)

# -------------------------
# 2. Iterate sorted timestamps
# -------------------------
sorted_timestamps = sorted(trajectories.keys())

for timestamp in sorted_timestamps:
    # pose 없는 경우 skip
    if TARGET_CAM not in trajectories[timestamp]:
        continue

    # 이미지 없는 경우 skip
    if timestamp not in records_camera:
        continue
    if TARGET_CAM not in records_camera[timestamp]:
        continue

    # 이미지 경로
    rel_path = records_camera[timestamp][TARGET_CAM]
    img_path = os.path.join(root, "sensors", "records_data", rel_path)

    if not os.path.exists(img_path):
        print(f"[WARN] Missing image: {img_path}")
        continue

    pose_wc = trajectories[timestamp][TARGET_CAM]  # world→cam pose

    # -------------------------
    # 2-1. 카메라 월드 위치 = pose.inverse()
    # -------------------------
    pose_cw = pose_wc.inverse()  # camera→world
    cam_pos = pose_cw.t.flatten()  # (x, y, z)

    xs.append(cam_pos[0])
    ys.append(cam_pos[1])
    zs.append(cam_pos[2])

    node_ids.append(node_id)
    image_paths.append(img_path)
    valid_timestamps.append(timestamp)

    node_id += 1
    count += 1
    if count >= 500:
        break

xs = np.array(xs, dtype=float)
ys = np.array(ys, dtype=float)
zs = np.array(zs, dtype=float)

# -------------------------
# 3. Plot camera trajectory + orientation
# -------------------------
plt.figure(figsize=(12, 10))
colors = np.linspace(0, 1, len(xs))

plt.scatter(xs, ys, c=colors, cmap='gist_rainbow', s=40)
plt.plot(xs, ys, color='gray', linewidth=0.4, alpha=0.5)

for i, timestamp in enumerate(valid_timestamps):

    pose_wc = trajectories[timestamp][TARGET_CAM]
    R_wc = quaternion.as_rotation_matrix(pose_wc.r)      # world→camera
    R_cw = R_wc.T                                        # camera→world

    # 카메라 forward direction in world coords (+Z in camera frame)
    forward_world = R_cw @ np.array([0, 0, 1], float)

    scale = 0.1
    fx = forward_world[0] * scale
    fy = forward_world[1] * scale

    x = xs[i]
    y = ys[i]

    plt.arrow(
        x, y, fx, fy,
        head_width=0.1,
        head_length=0.1,
        fc='black',
        ec='black'
    )

    plt.text(
        x, y, str(node_ids[i]),
        fontsize=8,
        ha='center',
        va='bottom',
        color='black'
    )

plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.title(f"Trajectory + Orientation — {TARGET_CAM}")
plt.axis("equal")
plt.grid(True, alpha=0.3)

plt.savefig("trajectory_target_cam_with_orientation.png", dpi=300)
plt.close()

print("Trajectory saved to: trajectory_target_cam_with_orientation.png")


# -------------------------
# 4. Save extracted images
# -------------------------
# saved_count = 0
# for idx, img_path in enumerate(image_paths):
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"[WARN] Failed to load: {img_path}")
#         continue
#     save_path = os.path.join(save_img_dir, f"{idx:04d}.jpg")
#     cv2.imwrite(save_path, img)
#     saved_count += 1

# print(f"Saved {saved_count} images to: {save_img_dir}/")
