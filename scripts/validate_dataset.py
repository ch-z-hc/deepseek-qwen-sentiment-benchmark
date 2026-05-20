#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter
from pathlib import Path

REQUIRED_FIELDS = {"text", "domain", "label"}
ALLOWED_LABELS = {"负面", "中性", "正面"}

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def validate_file(path: Path):
    data = load_json(path)
    errors = []
    seen_texts = set()
    label_counter = Counter()
    domain_counter = Counter()
    duplicate_texts = 0

    for i, row in enumerate(data):
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            errors.append(f"{path.name}[{i}] missing fields: {sorted(missing)}")
            continue

        text = str(row["text"]).strip()
        domain = str(row["domain"]).strip()
        label = str(row["label"]).strip()

        if not text:
            errors.append(f"{path.name}[{i}] empty text")
        if not domain:
            errors.append(f"{path.name}[{i}] empty domain")
        if label not in ALLOWED_LABELS:
            errors.append(f"{path.name}[{i}] invalid label: {label}")

        if text in seen_texts:
            duplicate_texts += 1
        seen_texts.add(text)

        label_counter[label] += 1
        domain_counter[domain] += 1

    return {
        "file": str(path),
        "rows": len(data),
        "labels": dict(label_counter),
        "domains": dict(domain_counter),
        "duplicate_texts": duplicate_texts,
        "errors": errors[:50],
        "num_errors": len(errors),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/deepseek_hard")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results = []
    for name in ["train.json", "test.json"]:
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        results.append(validate_file(path))

    print(json.dumps(results, ensure_ascii=False, indent=2))

    total_errors = sum(x["num_errors"] for x in results)
    if total_errors:
        raise SystemExit(f"dataset validation failed: {total_errors} errors")

if __name__ == "__main__":
    main()