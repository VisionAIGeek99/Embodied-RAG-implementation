import os
import argparse
import yaml
from datetime import datetime

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

    # ---- config 생성 ----
    cfg = {
        "dataset": {
            "name": args.name,
            "raw_root": args.raw_path,
            "processed_root": processed_root,
            "target_camera": args.target_cam,
            "max_nodes": args.max_nodes
        }
    }

    # 기존 config 백업
    if os.path.exists(config_path):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = config_path + f".backup_{ts}"
        os.rename(config_path, backup)
        print(f"[INFO] 기존 구성 파일 백업: {backup}")

    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    print("=====================================")
    print(f" Dataset setup 완료!")
    print(f" Raw      : {args.raw_path}")
    print(f" Processed: {processed_root}")
    print(f" Config   : {config_path}")
    print("=====================================")

if __name__ == "__main__":
    main()
