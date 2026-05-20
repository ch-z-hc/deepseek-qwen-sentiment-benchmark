#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


def setup_local_cache(root: Path):
    cache_root = root / ".cache"
    os.environ.setdefault("HF_HOME", str(cache_root / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "hf_datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_instruction_prompt(text: str, domain: str) -> str:
    return (
        "你是一个中文短评论情感分析助手。"
        "请判断下面评论的情感，只能输出三个标签之一：正面、负面、中性。\n"
        f"领域：{domain}\n"
        f"评论：{text}\n"
        "情感："
    )


def to_supervised_example(example, tokenizer, max_length: int):
    prompt = build_instruction_prompt(example["text"], example["domain"])
    answer = example["label"]

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        attention_mask = attention_mask[:max_length]

    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len
        attention_mask += [0] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.environ.get("QWEN3_MODEL_PATH", "./models/Qwen3-8B"))
    parser.add_argument("--data_dir", type=str, default=os.environ.get("DATA_DIR", "data_deepseek_hard"))
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path.cwd()
    setup_local_cache(root)

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.json"
    test_path = data_dir / "test.json"

    if not train_path.exists():
        raise FileNotFoundError(f"找不到训练集：{train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"找不到测试集：{test_path}")

    train = load_json(train_path)
    test = load_json(test_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=os.environ.get("HF_LOCAL_FILES_ONLY", "1") != "0",
        use_fast=False,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = Dataset.from_list(train)
    test_ds = Dataset.from_list(test)

    def fn(ex):
        return to_supervised_example(ex, tokenizer, args.max_length)

    tokenized_train = train_ds.map(fn, remove_columns=train_ds.column_names)
    tokenized_test = test_ds.map(fn, remove_columns=test_ds.column_names)

    tokenized_train.save_to_disk(str(data_dir / "tokenized_train"))
    tokenized_test.save_to_disk(str(data_dir / "tokenized_test"))

    print(f"[DONE] train rows: {len(train)}")
    print(f"[DONE] test rows: {len(test)}")
    print(f"[DONE] saved {data_dir / 'tokenized_train'}")
    print(f"[DONE] saved {data_dir / 'tokenized_test'}")


if __name__ == "__main__":
    main()
