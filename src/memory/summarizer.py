# src/memory/summarizer.py

import os
import json
from openai import OpenAI
from src.utils.log_openai_usage import log_openai_usage

client = OpenAI()

PROMPT_PATH = "/disks/ssd1/kmw2622/workspace/embodied-rag/prompt/abstraction_prompt.txt"

SUMMARY_SAVE_DIR = "/disks/ssd1/kmw2622/workspace/embodied-rag/datasets/coex_1f_processed/summaries"
os.makedirs(SUMMARY_SAVE_DIR, exist_ok=True)


def load_prompt():
    with open(PROMPT_PATH, "r") as f:
        return f.read().strip()


def extract_summary_only(llm_output: str) -> str:
    """
    LLM이 JSON을 출력하든 plain text를 출력하든
    항상 Summary 문자열만 추출하는 함수.
    """
    # 1) JSON 형태일 수 있는 경우
    try:
        parsed = json.loads(llm_output)

        # JSON인 경우 → Summary 필드만 반환
        if isinstance(parsed, dict) and "Summary" in parsed:
            return parsed["Summary"]

        # Summary가 없으면 전체 문자열 반환
        return llm_output
    except Exception:
        # JSON 포맷이 아니면 그대로 반환
        return llm_output


def summarize_cluster(captions, cluster_name=None, save=False, max_len=600):
    """
    Summarize multiple captions into a high-level area summary.
    Always returns a pure summary string (never JSON).
    """

    if not captions:
        return ""

    # ---------------------------------------------------------
    # 1) 입력 길이 제한
    # ---------------------------------------------------------
    merged = " ".join(captions)
    if len(merged) > max_len:
        merged = " ".join(captions[:6] + captions[-4:])

    # ---------------------------------------------------------
    # 2) Prompt 준비
    # ---------------------------------------------------------
    template = load_prompt()
    prompt = template.replace("{environment descriptions}", merged)

    # ---------------------------------------------------------
    # 3) LLM 호출
    # ---------------------------------------------------------
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You abstract multiple environment descriptions into a single high-level summary."
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=160,
            temperature=0.0,
        )

        if log_openai_usage:
            try:
                log_openai_usage(response)
            except Exception:
                pass

        llm_output = response.choices[0].message.content.strip()

    except Exception as e:
        print("[Summarizer ERROR]:", e)
        llm_output = " ".join(captions[:3])

    # ---------------------------------------------------------
    # 4) JSON/문자열에 상관없이 Summary 문자열만 추출
    # ---------------------------------------------------------
    summary = extract_summary_only(llm_output)

    # ---------------------------------------------------------
    # 5) 저장 옵션
    # ---------------------------------------------------------
    if save and cluster_name is not None:
        out_path = os.path.join(SUMMARY_SAVE_DIR, f"{cluster_name}.json")
        with open(out_path, "w") as f:
            json.dump(
                {
                    "cluster_name": cluster_name,
                    "summary": summary,   # ← 문자열만 저장
                    "captions": captions,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[SUMMARY SAVED] {out_path}")

    return summary
