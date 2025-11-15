import csv
import os
from datetime import datetime

LOG_FILE = "/disks/ssd1/kmw2622/workspace/semantic_forest_generation/log/openai_api_usage_log.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "model", "prompt_tokens", "completion_tokens", "total_tokens",
            "prompt_cost_usd", "completion_cost_usd", "total_cost_usd", "user_prompt"
        ])

# per 1M token 단가 (standard tier 기준)
PRICES = {
    "gpt-5": {"prompt": 1.25 / 1_000_000, "completion": 10.00 / 1_000_000},
    "gpt-5-mini": {"prompt": 0.25 / 1_000_000, "completion": 2.00 / 1_000_000},
    "gpt-5-nano": {"prompt": 0.05 / 1_000_000, "completion": 0.40 / 1_000_000},
    "gpt-4.1": {"prompt": 2.00 / 1_000_000, "completion": 8.00 / 1_000_000},
    "gpt-4.1-mini": {"prompt": 0.40 / 1_000_000, "completion": 1.60 / 1_000_000},
    "gpt-4.1-nano": {"prompt": 0.10 / 1_000_000, "completion": 0.40 / 1_000_000},
    "gpt-4o": {"prompt": 2.50 / 1_000_000, "completion": 10.00 / 1_000_000},
    "gpt-4o-mini": {"prompt": 0.15 / 1_000_000, "completion": 0.60 / 1_000_000},
    "gpt-3.5-turbo": {"prompt": 0.50 / 1_000_000, "completion": 1.50 / 1_000_000},
}

def log_openai_usage(response, prompt=None):
    try:
        model = getattr(response, "model", "unknown")
        usage = getattr(response, "usage", None)

        if not usage:
            print("⚠️ usage 정보가 응답에 포함되지 않았습니다.")
            return

        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        # 모델별 단가 찾기
        matched = None
        for name in PRICES:
            if model.startswith(name):
                matched = PRICES[name]
                break
        price = matched or {"prompt": 0, "completion": 0}

        prompt_cost = prompt_tokens * price["prompt"]
        completion_cost = completion_tokens * price["completion"]
        total_cost = prompt_cost + completion_cost

        # CSV에 기록
        with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                model,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                round(prompt_cost, 6),
                round(completion_cost, 6),
                round(total_cost, 6),
            ])

        print(f"✅ {model} 기록 완료 — {total_tokens} tokens, ${total_cost:.6f}")

    except Exception as e:
        print(f"⚠️ 로그 저장 중 오류 발생: {e}")

