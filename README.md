# DeepSeek Teacher → Qwen3 Student: Chinese Sentiment Analysis with LoRA Distillation

This project uses **DeepSeek** as an external strong teacher model to build a multi-domain Chinese sentiment analysis **Hard Set**, then fine-tunes a local **Qwen3-8B** student model via **PEFT/LoRA**.

The focus is not simple sentiment classification, but verifying the stability of local LLMs on **weak sentiment, concessive expressions, and boundary samples**, and improving the local model's judgment through DeepSeek teacher data distillation.

## Final Results

DeepSeek-hard test set: 1,440 samples across 8 domains.

| Model / Annotator | Test Set | Accuracy |
|---|---|---|
| Base Qwen3-8B | DeepSeek-hard | 62.50% |
| DeepSeek-v4-pro (recheck) | DeepSeek-hard | 94.03% |
| **Qwen3-8B + LoRA (step 300)** | DeepSeek-hard | **98.61%** |

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

# Step 1: Install non-PyTorch dependencies first
# (requirements.txt does NOT include torch — install it separately below)
pip install -r requirements.txt

# Step 2: Install PyTorch according to YOUR CUDA/driver version
# Check your CUDA version first:
nvidia-smi

# CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.6:
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

#### CUDA Compatibility Check

If you see `torch.cuda.is_available() == False` or `The NVIDIA driver on your system is too old`, your PyTorch CUDA wheel may be newer than your NVIDIA driver supports. Verify with:

```bash
nvidia-smi

python -c "import torch; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```

If CUDA is unavailable, reinstall PyTorch with a compatible CUDA version:

```bash
pip uninstall -y torch torchvision torchaudio triton
pip install --no-cache-dir torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
```

> **Important:** Always install `requirements.txt` first, then install PyTorch separately. Installing torch via requirements.txt may overwrite a working CUDA-compatible version.

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
#   DEEPSEEK_BASE_URL=https://api.deepseek.com   # optional, for custom endpoints
#   QWEN3_MODEL_PATH=./models/Qwen3-8B
#   DATA_DIR=data/deepseek_hard           # full reproduction
#   DATA_DIR=data_deepseek_hard           # verify released LoRA only
#   HF_LOCAL_FILES_ONLY=1
```

Or export directly:

```bash
export DEEPSEEK_API_KEY="your_key"
export QWEN3_MODEL_PATH="/path/to/Qwen3-8B"

# For full reproduction (generate data via DeepSeek):
export DATA_DIR="data/deepseek_hard"

# Or for verifying the released LoRA only:
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
  --output_dir data/deepseek_hard \
  --model deepseek-v4-flash \
  --train_per_bucket 200 \
  --test_per_bucket 60
```

### 2. Validate Dataset

```bash
python scripts/validate_dataset.py --data_dir data/deepseek_hard
```

### 3. Tokenize

```bash
python scripts/tokenize_deepseek_data.py \
  --model_path $QWEN3_MODEL_PATH \
  --data_dir data/deepseek_hard \
  --max_length 256
```

### 4. Evaluate Base Qwen3-8B

```bash
python scripts/evaluate.py \
  --model_path $QWEN3_MODEL_PATH \
  --base_tokenizer_path $QWEN3_MODEL_PATH \
  --test_file data/deepseek_hard/test.json \
  --output_dir results/deepseek_hard/base_qwen3 \
  --device cuda:0 \
  --batch_size 4
```

### 5. DeepSeek-v4-pro External Recheck

```bash
python scripts/evaluate_deepseek_classifier.py \
  --test_file data/deepseek_hard/test.json \
  --output_dir results/deepseek_hard/deepseek_classifier_pro \
  --model deepseek-v4-pro \
  --batch_size 20
```

### 6. Train Qwen3-LoRA

```bash
python scripts/train_lora.py \
  --model_path $QWEN3_MODEL_PATH \
  --train_dataset data/deepseek_hard/tokenized_train \
  --eval_dataset data/deepseek_hard/tokenized_test \
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
  --test_file data/deepseek_hard/test.json \
  --output_dir results/deepseek_hard/lora_step300 \
  --device cuda:0 \
  --batch_size 4
```

## Verify Released LoRA (Without Regenerating Data)

If you only want to verify that the released LoRA adapter works — without calling DeepSeek, regenerating data, or training — follow this shorter workflow using the official release dataset.

### Which data directory to use

| Directory | Purpose |
|---|---|
| `data/deepseek_hard/` | Data you generate yourself by calling DeepSeek (full reproduction) |
| `data_deepseek_hard/` | Official release dataset — use this to verify the LoRA without regeneration |

The official dataset is available on the [Release page](https://github.com/ch-z-hc/deepseek-qwen-sentiment-benchmark/releases/tag/v1.0) as `deepseek_hard_dataset_json.tar.gz`.

```bash
# Download and extract the official dataset
tar -xzf deepseek_hard_dataset_json.tar.gz
# This creates data_deepseek_hard/
```

### Extract the LoRA adapter

Download `qwen3-8b-lora-deepseek-hard-step300.tar.gz` from the [Release page](https://github.com/ch-z-hc/deepseek-qwen-sentiment-benchmark/releases/tag/v1.0).

```bash
# Extract to a temp location first to check the internal structure
tar -tzf qwen3-8b-lora-deepseek-hard-step300.tar.gz | head -5

# If the tarball already contains a models/ prefix, extract to project root:
tar -xzf qwen3-8b-lora-deepseek-hard-step300.tar.gz
# → models/qwen3-8b-lora-deepseek-hard-step300/

# If the tarball does NOT have a models/ prefix, extract to ./models:
mkdir -p models
tar -xzf qwen3-8b-lora-deepseek-hard-step300.tar.gz -C ./models
# → models/qwen3-8b-lora-deepseek-hard-step300/

# Verify the adapter files are in the right place:
ls models/qwen3-8b-lora-deepseek-hard-step300/
# If not found, search for it:
find . -name adapter_config.json
```

> **Note:** If you end up with `models/models/qwen3-8b-lora-deepseek-hard-step300/`, the tarball had an internal `models/` prefix and you extracted it into `./models`. Just move it up one level: `mv models/models/qwen3-8b-lora-deepseek-hard-step300 models/`

### Verification steps

Set your environment:

```bash
export QWEN3_MODEL_PATH="/path/to/Qwen3-8B"
export DATA_DIR="data_deepseek_hard"
```

Then run:

```bash
# 1. Validate the official dataset
python scripts/validate_dataset.py --data_dir data_deepseek_hard

# 2. Tokenize
python scripts/tokenize_deepseek_data.py \
  --model_path $QWEN3_MODEL_PATH \
  --data_dir data_deepseek_hard \
  --max_length 256

# 3. Evaluate Base Qwen3-8B
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --model_path $QWEN3_MODEL_PATH \
  --base_tokenizer_path $QWEN3_MODEL_PATH \
  --test_file data_deepseek_hard/test.json \
  --output_dir results/deepseek_hard/base_qwen3_official_verify \
  --device cuda:0 \
  --batch_size 4

# 4. Evaluate LoRA adapter
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_lora.py \
  --base_model_path $QWEN3_MODEL_PATH \
  --lora_path ./models/qwen3-8b-lora-deepseek-hard-step300 \
  --test_file data_deepseek_hard/test.json \
  --output_dir results/deepseek_hard/lora_step300_official_verify \
  --device cuda:0 \
  --batch_size 4
```

Or use the one-click script:

```bash
export QWEN3_MODEL_PATH="/path/to/Qwen3-8B"
bash scripts/verify_released_lora.sh
```

## Understanding Results

### invalid_count

`invalid_count` indicates how many model outputs could not be parsed into one of the three valid labels: `负面 / 中性 / 正面`.

- Base Qwen3-8B may produce explanatory text or non-standard labels (e.g., synonyms, reasoning), resulting in some invalid outputs.
- These samples are counted as errors in accuracy calculations.
- After LoRA fine-tuning, the output format is significantly constrained, and `invalid_count` typically drops to 0.

A non-zero `invalid_count` for Base Qwen3-8B is expected and does not indicate a script failure.

### Expected Results

Due to minor differences in generation and parsing, Base Qwen3-8B accuracy may vary slightly from the reported 62.50%. A successful reproduction should show:

| Checkpoint | Expected Accuracy |
|---|---|
| Base Qwen3-8B | ~62% (e.g., 62.43% is fine) |
| Qwen3-8B + LoRA step300 | ~98.6% |
| LoRA invalid_count | 0 |

If your LoRA accuracy is near 98.6% and Base is around 62%, the reproduction is successful — small deviations (±0.2%) are normal.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── configs/                  # Label schema documentation
├── docs/                     # OS-specific reproduction guides
├── scripts/
│   ├── generate_deepseek_hard_data.py   # Data generation via DeepSeek
│   ├── validate_dataset.py              # Dataset integrity check
│   ├── tokenize_deepseek_data.py        # Tokenization
│   ├── evaluate.py                      # Evaluate base model
│   ├── evaluate_deepseek_classifier.py  # DeepSeek external recheck
│   ├── train_lora.py                    # LoRA fine-tuning
│   ├── evaluate_lora.py                 # Evaluate LoRA adapter
│   ├── auto_judge.py                    # LLM-as-Judge scoring
│   ├── check_env.py                     # Environment validation
│   ├── verify_released_lora.sh          # One-click verification
│   └── utils.py                         # Shared utilities
├── data/
│   └── deepseek_hard/        # Generated datasets (train/test/seed.json)
├── data_deepseek_hard/       # Official release dataset (from GitHub Release)
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

### Extraction notes

The tarball may have an internal `models/` prefix. Check before extracting:

```bash
# Preview the tarball structure:
tar -tzf qwen3-8b-lora-deepseek-hard-step300.tar.gz | head -5

# If the listing shows models/... as the top-level, extract to project root:
tar -xzf qwen3-8b-lora-deepseek-hard-step300.tar.gz

# If there is no models/ prefix, extract into ./models:
mkdir -p models
tar -xzf qwen3-8b-lora-deepseek-hard-step300.tar.gz -C ./models
```

Verify the adapter is at the correct path:

```bash
ls models/qwen3-8b-lora-deepseek-hard-step300/adapter_config.json
```

If you see `models/models/...`, move it up: `mv models/models/qwen3-8b-lora-deepseek-hard-step300 models/`

## Windows Users

See [docs/REPRODUCE_WINDOWS.md](docs/REPRODUCE_WINDOWS.md) for a detailed Windows-specific guide.
