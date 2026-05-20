#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import gc
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer


LABELS = ["负面", "中性", "正面"]
LABEL2ID = {"负面": 0, "中性": 1, "正面": 2}


def project_root() -> Path:
    return Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()


def setup_local_cache(root: Path) -> None:
    cache_root = root / ".cache"
    os.environ.setdefault("HF_HOME", str(cache_root / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "hf_datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_instruction_prompt(text: str, domain: str) -> str:
    return (
        "你是一个中文短评论情感分析助手。"
        "请判断下面评论的情感，只能输出三个标签之一：正面、负面、中性。\n"
        f"领域：{domain}\n"
        f"评论：{text}\n"
        "情感："
    )


def parse_label(text: str) -> str:
    """
    Strict label parser.

    Benchmark-friendly behavior:
    - Accept only explicit labels: 正面 / 负面 / 中性.
    - Do not guess with sentiment keywords like 好 / 差 / 喜欢.
    - If the model does not produce a clean label, mark it as INVALID.
    """
    s = re.sub(r"\s+", "", text.strip())
    head = s[:30]

    if head in LABEL2ID:
        return head

    matches = [label for label in ["正面", "负面", "中性"] if label in head]

    if len(matches) == 1:
        return matches[0]

    return "INVALID"

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


@torch.inference_mode()
def predict_batch(model, tokenizer, batch, device, max_new_tokens):
    prompts = [build_instruction_prompt(x["text"], x["domain"]) for x in batch]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    generated = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = generated[:, input_len:]
    outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    rows = []
    for item, raw in zip(batch, outputs):
        pred = parse_label(raw)
        gold = item["label"]
        rows.append(
            {
                "id": item.get("id"),
                "text": item["text"],
                "domain": item["domain"],
                "gold": gold,
                "prediction": pred,
                "raw_output": raw.strip(),
                "correct": int(pred == gold),
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default=os.environ.get("QWEN3_MODEL_PATH", "./models/Qwen3-8B"))
    parser.add_argument("--lora_path", type=str, default="./models/qwen3-8b-lora-deepseek-hard-step300")
    parser.add_argument("--test_file", type=str, default=os.path.join(os.environ.get("DATA_DIR", "data_deepseek_hard"), "test.json"))
    parser.add_argument("--output_dir", type=str, default="results/deepseek_hard/lora_step300")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    root = project_root()
    setup_local_cache(root)

    base_model_path = Path(args.base_model_path)
    lora_path = (root / args.lora_path).resolve() if not Path(args.lora_path).is_absolute() else Path(args.lora_path)
    test_file = (root / args.test_file).resolve() if not Path(args.test_file).is_absolute() else Path(args.test_file)
    output_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not base_model_path.exists():
        raise FileNotFoundError(f"基座模型不存在：{base_model_path}")
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA adapter 不存在：{lora_path}")
    if not test_file.exists():
        raise FileNotFoundError(f"测试集不存在：{test_file}")

    print(f"[INFO] Loading tokenizer from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
        local_files_only=os.environ.get("HF_LOCAL_FILES_ONLY", "1") != "0",
        use_fast=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading base model from {base_model_path}")
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
        local_files_only=os.environ.get("HF_LOCAL_FILES_ONLY", "1") != "0",
        torch_dtype=dtype,
    ).to(args.device)

    print(f"[INFO] Loading LoRA adapter from {lora_path}")
    model = PeftModel.from_pretrained(base_model, str(lora_path))
    model.eval()

    test_data = load_json(test_file)
    predictions = []

    for idx, batch in enumerate(batched(test_data, args.batch_size), start=1):
        predictions.extend(
            predict_batch(model, tokenizer, batch, args.device, args.max_new_tokens)
        )
        if idx % 10 == 0:
            print(f"[INFO] evaluated {len(predictions)}/{len(test_data)}")

    y_true = [LABEL2ID[x["gold"]] for x in predictions]
    y_pred = [LABEL2ID.get(x["prediction"], -1) for x in predictions]
    invalid_count = sum(1 for x in predictions if x["prediction"] not in LABEL2ID)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2])
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=[0, 1, 2])
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
    for row in predictions:
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
        "base_model_path": str(base_model_path),
        "lora_path": str(lora_path),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "labels": LABELS,
        "confusion_matrix": cm,
        "classification_report": report,
        "per_domain": per_domain,
        "num_samples": len(predictions),
        "invalid_count": invalid_count,
        "invalid_rate": invalid_count / max(1, len(predictions)),
    }

    save_json(output_dir / "eval_results.json", results)
    write_jsonl(output_dir / "predictions.jsonl", predictions)

    print(json.dumps(results, ensure_ascii=False, indent=2))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
