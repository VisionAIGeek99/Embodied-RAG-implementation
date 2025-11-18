# scripts/02_caption_nodes.py

import os
import json
import yaml
import base64
import argparse

from src.utils.log_openai_usage import log_openai_usage

# ===========================
# API KEY
# ===========================

print("OPENAI_API_KEY loaded:", "OPENAI_API_KEY" in os.environ)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("[WARN] openai 패키지를 찾을 수 없습니다.")


# ===========================
# 사용량 로거 임포트
# ===========================
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


try:
    from src.utils.log_openai_usage import log_openai_usage
except Exception as e:
    print("[WARN] log_openai_usage 임포트 실패:", e)
    log_openai_usage = None


# ===========================
# Config & Prompt 로딩
# ===========================
def load_config():
    cfg_path = os.path.join(ROOT_DIR, "config", "dataset_config.yaml")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def load_prompt_from_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ prompt 파일 없음: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def encode_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ===========================
# OpenAI Caption 함수
# ===========================
def generate_caption_with_openai(client, model, image_path, caption_prompt):
    img_b64 = encode_image_b64(image_path)
    image_url = f"data:image/jpeg;base64,{img_b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": caption_prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )

    # 로그 기록
    if log_openai_usage:
        try:
            log_openai_usage(resp)
        except Exception as e:
            print("[WARN] usage logging 실패:", e)

    return resp.choices[0].message.content.strip()


# ===========================
# MAIN
# ===========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_nodes", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = load_config()

    processed_root = os.path.join(ROOT_DIR, "datasets", "coex_1f_processed")
    nodes_raw_path = os.path.join(processed_root, "nodes_raw.json")
    nodes_out_path = os.path.join(processed_root, "nodes_with_captions.json")

    # prompt 파일 경로
    caption_prompt_path = os.path.join(ROOT_DIR, "prompt", "caption_prompt.txt")
    caption_prompt = load_prompt_from_file(caption_prompt_path)

    if not os.path.exists(nodes_raw_path):
        raise FileNotFoundError(f"❌ nodes_raw.json 없음: {nodes_raw_path}")

    print(f"[INFO] nodes_raw.json 로딩: {nodes_raw_path}")
    with open(nodes_raw_path, "r") as f:
        nodes = json.load(f)

    if len(nodes) == 0:
        print("[INFO] 노드 없음, 종료")
        return

    client = OpenAI() if (not args.dry_run and OpenAI is not None) else None
    max_n = args.max_nodes if args.max_nodes else len(nodes)

    # ===========================
    # CAPTION LOOP
    # ===========================
    for i, node in enumerate(nodes):
        if i >= max_n:
            break

        node_id = node["node_id"]
        img_path = node.get("image")

        if not img_path or not os.path.exists(img_path):
            print(f"[WARN] node {node_id}: 이미지 없음")
            node["caption"] = None
            continue

        if args.dry_run:
            node["caption"] = f"[DUMMY CAPTION] {os.path.basename(img_path)}"
            print(f"[DRY RUN] node {node_id} 더미 캡션 생성")
            continue

        print(f"[INFO] node {node_id}: 캡션 생성 중...")

        try:
            caption = generate_caption_with_openai(client, args.model, img_path, caption_prompt)
        except Exception as e:
            print(f"[ERROR] node {node_id}: caption 실패 → {e}")
            caption = None

        node["caption"] = caption

    # ===========================
    # Save 결과
    # ===========================
    with open(nodes_out_path, "w", encoding="utf-8") as f:
        json.dump(nodes, f, indent=2, ensure_ascii=False)

    print(f"[DONE] nodes_with_captions.json 저장 → {nodes_out_path}")


if __name__ == "__main__":
    main()
