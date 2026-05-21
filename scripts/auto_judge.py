#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 4: LLM-as-Judge scoring.

The judge model checks whether the model prediction is correct compared with
the human/gold sentiment label, then Cohen's Kappa is calculated between:
    - human_correct: prediction == gold label
    - judge_correct: judge says correct / wrong

Run:
    python scripts/auto_judge.py \
      --judge_model_path ./models/Qwen3-8B \
      --predictions_file results/predictions.jsonl \
      --device cuda:0
"""

import argparse
import gc
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.utils import (
    batched,
    project_root,
    read_jsonl,
    save_json,
    setup_local_cache,
    write_jsonl,
)


def build_judge_prompt(row: Dict) -> str:
    return (
        "你是严谨的中文情感分析评测员。请判断模型预测是否正确。\n"
        "规则：如果模型预测的情感与人工标签一致，或者语义上等价，则输出“正确”；否则输出“错误”。\n"
        "只能输出一个词：正确 或 错误。\n\n"
        "示例1：评论：太香了吧，下次还来！ 人工标签：正面 模型预测：正面 判断：正确\n"
        "示例2：评论：真的踩雷了 人工标签：负面 模型预测：正面 判断：错误\n\n"
        f"领域：{row.get('domain', '')}\n"
        f"评论：{row.get('text', '')}\n"
        f"人工标签：{row.get('gold', '')}\n"
        f"模型预测：{row.get('prediction', '')}\n"
        "判断："
    )


def parse_judge_output(text: str) -> int:
    s = re.sub(r"\s+", "", text.strip())
    head = s[:20]
    if "正确" in head and "错误" not in head:
        return 1
    if "错误" in head or "不正确" in head or "错" in head:
        return 0
    # Fallback: exact label agreement is a conservative default.
    return -1


@torch.inference_mode()
def judge_batch(model, tokenizer, batch: List[Dict], device: str, max_new_tokens: int) -> List[Dict]:
    prompts = [build_judge_prompt(x) for x in batch]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=768,
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

    judged = []
    for row, raw in zip(batch, outputs):
        human_correct = int(row.get("prediction") == row.get("gold"))
        judge_correct = parse_judge_output(raw)
        # Mark unparseable as None so they are excluded from Kappa
        judged.append(
            {
                **row,
                "human_correct": human_correct,
                "judge_correct": None if judge_correct == -1 else int(judge_correct),
                "judge_raw_output": raw.strip(),
            }
        )
    return judged


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge_model_path", type=str, default=os.environ.get("QWEN3_MODEL_PATH", "./models/Qwen3-8B"))
    parser.add_argument("--predictions_file", type=str, default="results/predictions.jsonl")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=6)
    parser.add_argument("--allow_fallback_device", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = project_root()
    setup_local_cache(root)

    judge_model_path = Path(args.judge_model_path)
    pred_file = (root / args.predictions_file).resolve() if not Path(args.predictions_file).is_absolute() else Path(args.predictions_file)
    output_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not judge_model_path.exists():
        raise FileNotFoundError(f"评判模型路径不存在：{judge_model_path}")
    if not pred_file.exists():
        raise FileNotFoundError(f"预测文件不存在：{pred_file}，请先运行 scripts/evaluate.py")

    device = args.device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            if args.allow_fallback_device:
                device = "cpu"
                print("[WARN] CUDA unavailable, fallback to CPU")
            else:
                raise RuntimeError("未检测到 CUDA，但当前 device 参数为 CUDA。")
        else:
            idx = int(device.split(":")[1]) if ":" in device else 0
            if idx >= torch.cuda.device_count():
                if args.allow_fallback_device:
                    device = "cuda:0"
                    print(f"[WARN] {args.device} unavailable, fallback to cuda:0")
                else:
                    raise RuntimeError(f"{args.device} 不可用，当前只有 {torch.cuda.device_count()} 张 GPU。")

    print(f"[INFO] Loading judge tokenizer from {judge_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(judge_model_path),
        trust_remote_code=True,
        local_files_only=os.environ.get("HF_LOCAL_FILES_ONLY", "1") != "0",
        use_fast=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading judge model from {judge_model_path} on {device}")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(judge_model_path),
        trust_remote_code=True,
        local_files_only=os.environ.get("HF_LOCAL_FILES_ONLY", "1") != "0",
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    rows = read_jsonl(pred_file)
    details = []
    for batch in tqdm(batched(rows, args.batch_size), desc="Judging",
                      total=(len(rows) + args.batch_size - 1) // args.batch_size):
        details.extend(judge_batch(model, tokenizer, batch, device, args.max_new_tokens))

    # Exclude unparseable judge outputs from Kappa calculation
    valid = [x for x in details if x["judge_correct"] is not None]
    num_unparseable = len(details) - len(valid)

    human = [x["human_correct"] for x in valid]
    judge = [x["judge_correct"] for x in valid]

    kappa = cohen_kappa_score(human, judge)
    judge_acc_against_human = accuracy_score(human, judge)

    result = {
        "num_samples": len(details),
        "num_unparseable": num_unparseable,
        "cohen_kappa": kappa,
        "judge_accuracy_against_human_correctness": judge_acc_against_human,
        "human_correct_rate": sum(human) / max(1, len(human)),
        "judge_correct_rate": sum(judge) / max(1, len(judge)),
    }

    save_json(output_dir / "judge_results.json", result)
    write_jsonl(output_dir / "judge_details.jsonl", details)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
