#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 3: Evaluate the fine-tuned sentiment classifier.

Run:
    python -m scripts.evaluate \
      --model_path ./models/qwen3-8b-finetuned \
      --test_file data/test.json \
      --device cuda:0
"""

import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.utils import (
    LABEL2ID,
    LABELS,
    batched,
    build_instruction_prompt,
    load_json,
    parse_label,
    project_root,
    save_json,
    setup_local_cache,
    write_jsonl,
)


@torch.inference_mode()
def predict_batch(model, tokenizer, batch: List[Dict], device: str, max_new_tokens: int) -> List[Dict]:
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
                "prediction": pred or "INVALID",
                "raw_output": raw.strip(),
                "correct": int(pred == gold),
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.environ.get("QWEN3_MODEL_PATH", "./models/Qwen3-8B"))
    parser.add_argument("--base_tokenizer_path", type=str, default=os.environ.get("QWEN3_MODEL_PATH", "./models/Qwen3-8B"))
    parser.add_argument("--test_file", type=str, default=os.path.join(os.environ.get("DATA_DIR", "data/deepseek_hard"), "test.json"))
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    root = project_root()
    setup_local_cache(root)

    model_path = (root / args.model_path).resolve() if not Path(args.model_path).is_absolute() else Path(args.model_path)
    test_file = (root / args.test_file).resolve() if not Path(args.test_file).is_absolute() else Path(args.test_file)
    output_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在：{model_path}")
    if not test_file.exists():
        raise FileNotFoundError(f"测试集不存在：{test_file}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA，但当前 device 参数为 CUDA。")

    tokenizer_path = model_path if (model_path / "tokenizer_config.json").exists() else Path(args.base_tokenizer_path)
    print(f"[INFO] Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        local_files_only=os.environ.get("HF_LOCAL_FILES_ONLY", "1") != "0",
        use_fast=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading model from {model_path} on {args.device}")
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=os.environ.get("HF_LOCAL_FILES_ONLY", "1") != "0",
        torch_dtype=dtype,
    ).to(args.device)
    model.eval()
    # Suppress warnings about sampling params ignored when do_sample=False
    for attr in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, attr):
            setattr(model.generation_config, attr, None)

    test_data = load_json(test_file)
    predictions = []
    for batch in tqdm(batched(test_data, args.batch_size), desc="Evaluating",
                      total=(len(test_data) + args.batch_size - 1) // args.batch_size):
        predictions.extend(
            predict_batch(model, tokenizer, batch, args.device, args.max_new_tokens)
        )

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
