#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def ok(name, value):
    print(f"[OK] {name}: {value}")

def warn(name, value):
    print(f"[WARN] {name}: {value}")

def fail(name, value):
    print(f"[FAIL] {name}: {value}")

def check_import(pkg, import_name=None):
    import_name = import_name or pkg
    try:
        mod = __import__(import_name)
        ok(pkg, getattr(mod, "__version__", "installed"))
        return True
    except Exception as e:
        fail(pkg, repr(e))
        return False

def main():
    print("== Python ==")
    ok("python", sys.version.replace("\n", " "))

    print("\n== Packages ==")
    for pkg, import_name in [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("peft", "peft"),
        ("scikit-learn", "sklearn"),
        ("openai", "openai"),
        ("matplotlib", "matplotlib"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
        ("python-dotenv", "dotenv"),
    ]:
        check_import(pkg, import_name)

    print("\n== CUDA ==")
    try:
        import torch
        ok("torch.cuda.is_available()", torch.cuda.is_available())
        if torch.cuda.is_available():
            ok("cuda device count", torch.cuda.device_count())
            ok("cuda device 0", torch.cuda.get_device_name(0))
        else:
            warn("cuda", "not available; evaluation can run on CPU slowly, LoRA training is not recommended")
    except Exception as e:
        fail("cuda check", repr(e))

    print("\n== Paths ==")
    qwen_path = Path(os.environ.get("QWEN3_MODEL_PATH", ""))
    if qwen_path and qwen_path.exists():
        ok("QWEN3_MODEL_PATH", qwen_path)
        for f in ["config.json", "tokenizer_config.json"]:
            p = qwen_path / f
            if p.exists():
                ok(f, p)
            else:
                warn(f, f"missing under {qwen_path}")
    else:
        fail("QWEN3_MODEL_PATH", f"not found: {qwen_path}")

    data_dir = Path(os.environ.get("DATA_DIR", "data/deepseek_hard"))
    if data_dir.exists():
        ok("DATA_DIR", data_dir)
        for f in ["train.json", "test.json"]:
            p = data_dir / f
            if p.exists():
                ok(f, p)
            else:
                warn(f, f"missing under {data_dir}")
    else:
        warn("DATA_DIR", f"not found yet: {data_dir}")

    print("\n== API Key ==")
    if os.environ.get("DEEPSEEK_API_KEY"):
        ok("DEEPSEEK_API_KEY", "set")
    else:
        warn("DEEPSEEK_API_KEY", "not set; DeepSeek data generation/recheck will fail")

if __name__ == "__main__":
    main()