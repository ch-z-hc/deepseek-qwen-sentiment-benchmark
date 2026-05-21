#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import httpx
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from scripts.utils import (
    LABEL2ID,
    LABELS,
    append_jsonl,
    batched,
    extract_json_array,
    load_json,
    project_root,
    read_jsonl,
    save_json,
)


def normalize_label(x: str):
    s = re.sub(r"\s+", "", str(x or ""))
    if "正面" in s or "积极" in s or "正向" in s:
        return "正面"
    if "负面" in s or "消极" in s or "负向" in s:
        return "负面"
    if "中性" in s or "中立" in s or "客观" in s:
        return "中性"
    return None


def build_messages(batch: List[Dict]):
    system = (
        "你是严格的中文短评论情感分类器。"
        "你需要根据评论整体语义判断情感标签，只能在“负面、中性、正面”中选择一个。"
        "不要被局部词语误导，要判断整句话的整体态度。"
        "输出必须是 JSON 数组，不要解释。"
    )

    items = []
    for x in batch:
        items.append({
            "id": x["id"],
            "domain": x.get("domain", ""),
            "text": x.get("text", "")
        })

    user = f"""
请对下面每条评论进行三分类情感判断。

标签定义：
- 负面：整体表达不满、抱怨、失望、不推荐、体验差。
- 中性：整体是事实陈述、普通体验、弱情绪或态度不明显；没有明确推荐或抱怨。
- 正面：整体表达满意、认可、喜欢、推荐、体验好。

注意：
1. 如果句子有转折，请按整体最终态度判断。
2. 不要输出解释。
3. 必须保持原 id。
4. 只输出 JSON 数组。

输入：
{json.dumps(items, ensure_ascii=False)}

输出格式：
[
  {{"id": 0, "label": "负面/中性/正面"}}
]
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_deepseek(client, model: str, messages: List[Dict], max_retries: int = 5):
    use_extra = model in {"deepseek-v4-flash", "deepseek-v4-pro"}
    last_err = None
    for attempt in range(max_retries):
        try:
            kwargs = {}
            if use_extra:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                top_p=1,
                max_tokens=2048,
                stream=False,
                **kwargs,
            )
            content = resp.choices[0].message.content
            return extract_json_array(content)

        except Exception as e:
            last_err = e
            # If extra_body caused the failure, disable it for subsequent retries
            if use_extra and "extra_body" in str(e).lower():
                use_extra = False

            wait = min(2 ** attempt, 30)
            print(f"[WARN] DeepSeek call failed: {last_err}; retry in {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"DeepSeek failed after retries: {last_err}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_file", type=str, default="data/deepseek_hard/test.json")
    p.add_argument("--output_dir", type=str, default="results/deepseek_hard/deepseek_classifier")
    p.add_argument("--model", type=str, default="deepseek-v4-flash")
    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--sleep", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("请先 export DEEPSEEK_API_KEY")

    test_file = Path(args.test_file)
    output_dir = Path(args.output_dir)
    pred_file = output_dir / "predictions.jsonl"
    result_file = output_dir / "eval_results.json"

    data = read_json(test_file)
    if args.limit > 0:
        data = data[:args.limit]

    # Ensure ids are stable.
    for i, x in enumerate(data):
        if "id" not in x:
            x["id"] = i

    done_rows = read_jsonl(pred_file)
    done_ids = {x["id"] for x in done_rows}

    client = OpenAI(
        api_key=key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        timeout=httpx.Timeout(600.0, connect=60.0),
    )

    remain = [x for x in data if x["id"] not in done_ids]
    print(f"[INFO] total={len(data)}, done={len(done_ids)}, remain={len(remain)}")

    total_batches = (len(remain) + args.batch_size - 1) // args.batch_size
    for batch in tqdm(batched(remain, args.batch_size), desc="DeepSeek classify",
                      total=total_batches):
        arr = call_deepseek(client, args.model, build_messages(batch))

        pred_by_id = {}
        for item in arr:
            rid = item.get("id")
            label = normalize_label(item.get("label"))
            if rid is not None and label in LABEL2ID:
                pred_by_id[rid] = label

        out_rows = []
        for x in batch:
            pred = pred_by_id.get(x["id"])
            if pred is None:
                pred = "UNKNOWN"

            gold = x["label"]
            out_rows.append({
                "id": x["id"],
                "text": x["text"],
                "domain": x["domain"],
                "gold": gold,
                "prediction": pred,
                "correct": int(pred == gold),
                "model": args.model,
            })

        append_jsonl(pred_file, out_rows)
        time.sleep(args.sleep)

    rows = read_jsonl(pred_file)
    # If previous full output exists and now running limit, keep only current test ids.
    valid_ids = {x["id"] for x in data}
    rows = [x for x in rows if x["id"] in valid_ids]

    # Track unknown predictions
    unknown_count = sum(1 for x in rows if x["prediction"] == "UNKNOWN")

    y_true = [LABEL2ID[x["gold"]] for x in rows]
    y_pred = [LABEL2ID.get(x["prediction"], -1) for x in rows]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    domain_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in rows:
        d = row["domain"]
        domain_stats[d]["total"] += 1
        domain_stats[d]["correct"] += int(row["correct"])

    per_domain = {
        d: {
            "total": v["total"],
            "correct": v["correct"],
            "accuracy": v["correct"] / max(1, v["total"]),
        }
        for d, v in sorted(domain_stats.items())
    }

    results = {
        "model": args.model,
        "test_file": str(test_file),
        "num_samples": len(rows),
        "unknown_count": unknown_count,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "labels": LABELS,
        "confusion_matrix": cm,
        "classification_report": report,
        "per_domain": per_domain,
    }

    save_json(result_file, results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
