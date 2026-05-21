#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType

from scripts.utils import project_root, setup_local_cache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.environ.get("QWEN3_MODEL_PATH", "./models/Qwen3-8B"))
    parser.add_argument("--train_dataset", type=str, default=os.path.join(os.environ.get("DATA_DIR", "data/deepseek_hard"), "tokenized_train"))
    parser.add_argument("--eval_dataset", type=str, default=os.path.join(os.environ.get("DATA_DIR", "data/deepseek_hard"), "tokenized_test"))
    parser.add_argument("--output_dir", type=str, default="./models/qwen3-8b-lora")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume training from")
    return parser.parse_args()


def main():
    args = parse_args()
    root = project_root()
    setup_local_cache(root)

    model_path = Path(args.model_path)
    train_dataset_path = root / args.train_dataset
    eval_dataset_path = root / args.eval_dataset
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=os.environ.get("HF_LOCAL_FILES_ONLY", "1") != "0",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=os.environ.get("HF_LOCAL_FILES_ONLY", "1") != "0",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = load_from_disk(str(train_dataset_path))
    eval_dataset = load_from_disk(str(eval_dataset_path))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[INFO] LoRA adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
