#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from scripts.utils import (
    build_instruction_prompt,
    load_json,
    project_root,
    setup_local_cache,
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
    parser.add_argument("--data_dir", type=str, default=os.environ.get("DATA_DIR", "data/deepseek_hard"))
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    root = project_root()
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
