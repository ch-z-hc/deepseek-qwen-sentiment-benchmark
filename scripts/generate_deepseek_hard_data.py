#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List

import httpx
from openai import OpenAI

from scripts.utils import extract_json_array, save_json

DOMAINS = [
    "美食", "旅行", "购物", "娱乐",
    "教育", "医疗健康", "数码科技", "运动健身",
]

SENTIMENTS = ["负面", "中性", "正面"]
LABEL2ID = {"负面": 0, "中性": 1, "正面": 2}


def normalize_text(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"\s+", "", x)
    return x.strip(" -—\"'“”‘’`[]()（）")


def valid_text(x: str) -> bool:
    if not (6 <= len(x) <= 80):
        return False
    if not re.search(r"[\u4e00-\u9fff]", x):
        return False
    bad = ["领域：", "情感：", "标签：", "以下是", "无法", "不能", "JSON"]
    if any(b in x for b in bad):
        return False
    return True


def build_prompt(domain: str, sentiment: str, n: int, split: str) -> List[Dict]:
    system = (
        "你是中文情感分析数据集专家。"
        "你需要生成高质量 hard case 短评论，用来区分中性、弱正面、弱负面。"
        "输出必须是 JSON 数组，不要解释。"
    )

    if sentiment == "中性":
        rule = (
            "中性样本要求：必须是事实陈述、普通体验、轻微描述或信息记录；"
            "不能出现明显推荐、喜欢、失望、抱怨、踩雷、太棒、太差等强情绪；"
            "可以包含'一般/还行/正常/普通/可以/目前/暂时'，但整体不能偏正或偏负。"
        )
        examples = [
            "包装基本完整，物流比预计晚了一天",
            "课程内容比较基础，适合先了解一下",
            "这家店位置挺好，菜量属于正常水平",
        ]
    elif sentiment == "负面":
        rule = (
            "负面样本要求：可以有一点优点或客观描述，但整体必须明显不满、抱怨、失望或不推荐；"
            "避免只写身体正常反应或普通事实，必须体现负向评价。"
        )
        examples = [
            "风景还可以，可排队体验太糟糕了",
            "客服回复倒是快，就是问题完全没解决",
            "味道不是不能吃，但确实不会再点了",
        ]
    else:
        rule = (
            "正面样本要求：可以有一点小缺点或转折，但整体必须明显满意、认可、推荐或喜欢；"
            "不能只是客观描述，必须体现正向评价。"
        )
        examples = [
            "价格不算便宜，但体验确实值这个价",
            "人多了一点，不过景色真的很惊艳",
            "刚开始不适应，后面越学越有收获",
        ]

    user = f"""
请生成 {n} 条中文 hard case 情感分析短评论。

领域：{domain}
目标标签：{sentiment}
数据划分：{split}

核心要求：
1. 每条 6 到 80 个中文字符。
2. 样本要比普通情感评论更难，允许转折、弱情绪、边界表达。
3. {rule}
4. 不要生成和示例完全相同的句子。
5. 每条必须严格对应目标标签。
6. 只输出 JSON 数组，不要 markdown，不要解释。

参考示例：
{examples}

JSON 格式：
[
  {{"text": "评论内容", "domain": "{domain}", "label": "{sentiment}"}}
]
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_deepseek(client, model: str, messages: List[Dict], max_retries=5):
    last = None
    for i in range(max_retries):
        try:
            kwargs = {}
            if model in {"deepseek-v4-flash", "deepseek-v4-pro"}:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.9,
                top_p=0.95,
                max_tokens=4096,
                stream=False,
                **kwargs,
            )
            return extract_json_array(resp.choices[0].message.content)
        except Exception as e:
            last = e
            wait = min(2 ** i, 30)
            print(f"[WARN] {e}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(last)


def generate_split(client, args, split: str, per_bucket: int):
    rows = []
    seen = set()

    for domain in DOMAINS:
        for label in SENTIMENTS:
            got = 0
            attempts = 0
            print(f"[INFO] {split} {domain} {label} target={per_bucket}")
            while got < per_bucket and attempts < args.max_attempts_per_bucket:
                need = per_bucket - got
                n = min(args.request_batch_size, need + 10)
                arr = call_deepseek(client, args.model, build_prompt(domain, label, n, split))
                attempts += 1

                for item in arr:
                    text = normalize_text(item.get("text", ""))
                    d = item.get("domain", domain)
                    y = item.get("label", label)

                    if d != domain or y != label:
                        continue
                    if not valid_text(text):
                        continue

                    key = (split, domain, label, text)
                    if key in seen:
                        continue
                    seen.add(key)

                    rows.append({
                        "id": len(rows),
                        "text": text,
                        "domain": domain,
                        "label": label,
                        "label_id": LABEL2ID[label],
                        "source": "deepseek_hard_generated",
                        "split": split,
                    })
                    got += 1
                    if got >= per_bucket:
                        break

                print(f"  progress {got}/{per_bucket}, attempts={attempts}")
                time.sleep(args.sleep)

            if got < per_bucket:
                print(f"[WARN] not full: {split} {domain} {label} {got}/{per_bucket}")

    random.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = i
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="data/deepseek_hard")
    p.add_argument("--model", type=str, default="deepseek-v4-flash")
    p.add_argument("--train_per_bucket", type=int, default=200)
    p.add_argument("--test_per_bucket", type=int, default=60)
    p.add_argument("--request_batch_size", type=int, default=30)
    p.add_argument("--max_attempts_per_bucket", type=int, default=40)
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("请先 export DEEPSEEK_API_KEY")

    client = OpenAI(
        api_key=key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        timeout=httpx.Timeout(600.0, connect=60.0),
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = generate_split(client, args, "train", args.train_per_bucket)
    test = generate_split(client, args, "test", args.test_per_bucket)

    save_json(out / "train.json", train)
    save_json(out / "test.json", test)
    save_json(out / "seed.json", train + test)

    print(f"[DONE] train={len(train)}, test={len(test)}, total={len(train)+len(test)}")


if __name__ == "__main__":
    main()
