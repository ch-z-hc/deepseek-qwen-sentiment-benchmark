#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shared utilities for the deepseek-qwen-sentiment-benchmark project."""

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

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


# ---------------------------------------------------------------------------
# JSON / JSONL I/O
# ---------------------------------------------------------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    """Overwrite path with newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: List[Dict]) -> None:
    """Append rows to an existing JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------

def batched(items: List, batch_size: int) -> Iterator[List]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_instruction_prompt(text: str, domain: str) -> str:
    """Build the standard instruction prompt for sentiment classification."""
    return (
        "你是一个中文短评论情感分析助手。"
        "请判断下面评论的情感，只能输出三个标签之一：正面、负面、中性。\n"
        f"领域：{domain}\n"
        f"评论：{text}\n"
        "情感："
    )


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

def parse_label(text: str) -> Optional[str]:
    """
    Strict label parser.

    Accepts only explicit Chinese labels 正面 / 负面 / 中性.
    Returns the label string or None if unparseable.
    """
    s = re.sub(r"\s+", "", str(text or "").strip())
    head = s[:30]

    if head in LABEL2ID:
        return head

    matches = [label for label in LABELS if label in head]
    if len(matches) == 1:
        return matches[0]

    return None


# ---------------------------------------------------------------------------
# JSON extraction (shared by DeepSeek-generation scripts)
# ---------------------------------------------------------------------------

def extract_json_array(text: str):
    """Parse a JSON array from LLM output, with tolerant cleanup."""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("data"), list):
            return obj["data"]
    except Exception:
        pass

    m = re.search(r"\[.*\]", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    raise ValueError(f"Cannot parse JSON array from: {text[:500]}")
