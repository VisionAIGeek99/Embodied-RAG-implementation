import os
import argparse
import yaml
from datetime import datetime


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_sensors_txt(sensor_file, target_cam):
    """
    sensors.txt (CSV style) parsing:
    sensor_id, name, sensor_type, model, width, height, fx, fy, cx, cy, k1, k2, p1, p2
    """
    if not os.path.exists(sensor_file):
        raise FileNotFoundError(f"sensors.txt not found: {sensor_file}")

    cam_data = None

    with open(sensor_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("#") or len(line) == 0:
            continue

        parts = [p.strip() for p in line.split(",")]
        # print(parts)  # 디버깅 시 이용

        sensor_id = parts[0]
        if sensor_id != target_cam:
            continue

        sensor_type = parts[2]
        if sensor_type != "camera":
            continue

        model = parts[3]

        width = int(parts[4])
        height = int(parts[5])
        fx = float(parts[6])
        fy = float(parts[7])
        cx = float(parts[8])
        cy = float(parts[9])
        k1 = float(parts[10])
        k2 = float(parts[11])
        p1 = float(parts[12])
        p2 = float(parts[13])

        cam_data = {
            "camera_id": sensor_id,
            "sensor_type": sensor_type,
            "model": model,
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2
        }
        break

    if cam_data is None:
        raise ValueError(f"target camera {target_cam} not found in sensors.txt")

    return cam_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True,
                        help="Dataset name (ex: coex_1f)")
    parser.add_argument("--raw_path", type=str, required=True,
                        help="Raw kapture dataset root path")
    parser.add_argument("--target_cam", type=str, default="40027089_00")
    parser.add_argument("--max_nodes", type=int, default=1000)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(project_root, "config", "dataset_config.yaml")

    processed_root = os.path.join(project_root, "datasets", f"{args.name}_processed")

    # ---- 폴더 생성 ----
    ensure_dir(os.path.join(project_root, "config"))
    ensure_dir(os.path.join(project_root, "datasets"))
    ensure_dir(processed_root)
    ensure_dir(os.path.join(processed_root, "frames"))
    ensure_dir(os.path.join(processed_root, "viz"))

    # ---- sensors.txt 경로 ----
    sensor_file = os.path.join(
        args.raw_path,
        "sensors/sensors.txt"
    )

    camera_intrinsic = parse_sensors_txt(sensor_file, args.target_cam)

    # ---- config 생성 ----
    cfg = {
        "dataset": {
            "name": args.name,
            "raw_root": args.raw_path,
            "processed_root": processed_root,
            "target_camera": args.target_cam,
            "max_nodes": args.max_nodes,
            "camera": camera_intrinsic
        },
       
    }

    # 기존 config 백업
    if os.path.exists(config_path):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = config_path + f".backup_{ts}"
        os.rename(config_path, backup)
        print(f"[INFO] 기존 구성 파일 백업: {backup}")

    # ---- yaml 저장 ----
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    print("=====================================")
    print(f" Dataset setup 완료!")
    print(f" Raw      : {args.raw_path}")
    print(f" Processed: {processed_root}")
    print(f" Camera   : Loaded intrinsic for {args.target_cam}")
    print(f" Config   : {config_path}")
    print("=====================================")


if __name__ == "__main__":
    main()
