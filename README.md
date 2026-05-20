# DeepSeek Teacher → Qwen3 Student: Chinese Sentiment Analysis with LoRA Distillation

This project uses **DeepSeek** as an external strong teacher model to build a multi-domain Chinese sentiment analysis **Hard Set**, then fine-tunes a local **Qwen3-8B** student model via **PEFT/LoRA**.

The focus is not simple sentiment classification, but verifying the stability of local LLMs on **weak sentiment, concessive expressions, and boundary samples**, and improving the local model's judgment through DeepSeek teacher data distillation.

## Final Results

DeepSeek-hard test set: 1,440 samples across 8 domains.

| Model / Annotator | Test Set | Accuracy | Macro-F1 |
|---|---|---|---|
| Base Qwen3-8B | DeepSeek-hard | 62.50% | 59.48% |
| DeepSeek-v4-pro (recheck) | DeepSeek-hard | 94.03% | 93.89% |
| **Qwen3-8B + LoRA (step 300)** | DeepSeek-hard | **98.61%** | **98.61%** |

## Project Highlights

- Built a Chinese short-review Hard Set covering **8 domains × 3 sentiment classes** using DeepSeek as the teacher model.
- Hard Set focuses on weak sentiment, concessive expressions, and boundary samples.
- External recheck with DeepSeek-v4-pro achieved 94.03% accuracy, validating label consistency.
- Fine-tuned local Qwen3-8B with PEFT/LoRA (parameter-efficient), improving Hard Set accuracy from **62.50% → 98.61%**.
- Full pipeline is reproducible on both Linux and Windows.

## Quick Start

### 1. Setup Environment

```bash
conda create -n dsqwen python=3.10 -y
conda activate dsqwen

# Install PyTorch (choose according to your CUDA version)
# CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### 2. Download Qwen3-8B

The base model is required for both evaluation and LoRA training.

**Hugging Face (international):**

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B
```

**ModelScope (China, faster):**

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-8B --local_dir ./models/Qwen3-8B
```

**Git clone (if CLI tools are unavailable):**

```bash
# Hugging Face
git clone https://huggingface.co/Qwen/Qwen3-8B ./models/Qwen3-8B

# Or HF mirror (China)
git clone https://hf-mirror.com/Qwen/Qwen3-8B ./models/Qwen3-8B
```

> Model page: [Hugging Face](https://huggingface.co/Qwen/Qwen3-8B) | [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-8B)

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your settings:
#   DEEPSEEK_API_KEY=your_key
#   QWEN3_MODEL_PATH=./models/Qwen3-8B
#   DATA_DIR=data_deepseek_hard
#   HF_LOCAL_FILES_ONLY=1
```

Or export directly:

```bash
export DEEPSEEK_API_KEY="your_key"
export QWEN3_MODEL_PATH="/path/to/Qwen3-8B"
export DATA_DIR="data_deepseek_hard"
export HF_LOCAL_FILES_ONLY="1"
```

### 4. Verify Environment

```bash
python scripts/check_env.py
```

## Reproduce Full Pipeline

### 1. Generate DeepSeek Hard Set

```bash
python scripts/generate_deepseek_hard_data.py \
  --output_dir data_deepseek_hard \
  --model deepseek-v4-flash \
  --train_per_bucket 200 \
  --test_per_bucket 60
```

### 2. Validate Dataset

```bash
python scripts/validate_dataset.py --data_dir data_deepseek_hard
```

### 3. Tokenize

```bash
python scripts/tokenize_deepseek_data.py \
  --model_path $QWEN3_MODEL_PATH \
  --data_dir data_deepseek_hard \
  --max_length 256
```

### 4. Evaluate Base Qwen3-8B

```bash
python scripts/evaluate.py \
  --model_path $QWEN3_MODEL_PATH \
  --base_tokenizer_path $QWEN3_MODEL_PATH \
  --test_file data_deepseek_hard/test.json \
  --output_dir results/deepseek_hard/base_qwen3 \
  --device cuda:0 \
  --batch_size 4
```

### 5. DeepSeek-v4-pro External Recheck

```bash
python scripts/evaluate_deepseek_classifier.py \
  --test_file data_deepseek_hard/test.json \
  --output_dir results/deepseek_hard/deepseek_classifier_pro \
  --model deepseek-v4-pro \
  --batch_size 20
```

### 6. Train Qwen3-LoRA

```bash
python scripts/train_lora.py \
  --model_path $QWEN3_MODEL_PATH \
  --train_dataset data_deepseek_hard/tokenized_train \
  --eval_dataset data_deepseek_hard/tokenized_test \
  --output_dir ./models/qwen3-8b-lora-deepseek-hard-step300 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_steps 300
```

### 7. Evaluate Qwen3-LoRA

```bash
python scripts/evaluate_lora.py \
  --base_model_path $QWEN3_MODEL_PATH \
  --lora_path ./models/qwen3-8b-lora-deepseek-hard-step300 \
  --test_file data_deepseek_hard/test.json \
  --output_dir results/deepseek_hard/lora_step300 \
  --device cuda:0 \
  --batch_size 4
```

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── configs/                  # Label schema & config docs
├── docs/                     # Experiment results, reproduction guides
├── scripts/
│   ├── generate_deepseek_hard_data.py   # Step 1: Generate hard set via DeepSeek
│   ├── validate_dataset.py              # Step 1b: Validate dataset integrity
│   ├── tokenize_deepseek_data.py        # Step 2: Tokenize for Qwen3
│   ├── evaluate.py                      # Step 3: Evaluate base / fine-tuned model
│   ├── evaluate_deepseek_classifier.py  # Step 4: DeepSeek external recheck
│   ├── evaluate_lora.py                 # Step 5: Evaluate LoRA adapter
│   ├── train_lora.py                    # Step 6: LoRA fine-tuning
│   ├── auto_judge.py                    # LLM-as-Judge scoring
│   └── check_env.py                     # Environment validation
├── data/
│   └── deepseek_hard/        # Generated datasets (train/test/seed.json)
├── results/
│   └── deepseek_hard/        # Evaluation results
└── models/                   # LoRA adapter checkpoints
```

## Model Artifacts

The final LoRA adapter and project artifacts are available on GitHub Release:

https://github.com/ch-z-hc/deepseek-qwen-sentiment-benchmark/releases/tag/v1.0

Release assets:
- `qwen3-8b-lora-deepseek-hard-step300.tar.gz`
- `deepseek_hard_dataset_json.tar.gz`
- `final_results_deepseek_hard.tar.gz`

The base Qwen3-8B model is **not** included — download it separately from Hugging Face or ModelScope.

## Windows Users

See [docs/REPRODUCE_WINDOWS.md](docs/REPRODUCE_WINDOWS.md) for a detailed Windows-specific guide.

## Resume Summary

```
• Built a Chinese sentiment analysis Hard Set (8 domains, 3 classes, 1,440 samples) using DeepSeek as teacher model,
  focusing on weak sentiment, concessive expressions, and boundary samples.
• Fine-tuned local Qwen3-8B with PEFT/LoRA, improving Hard Set accuracy from 62.50% to 98.61%
  (Macro-F1: 59.48% → 98.61%).
• External recheck with DeepSeek-v4-pro achieved 94.03% accuracy, validating label consistency.
• Full pipeline is reproducible with a single conda environment and .env configuration.
```
