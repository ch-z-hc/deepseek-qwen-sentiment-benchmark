# Windows Reproduction Guide

## 1. Create environment

```powershell
conda create -n dsqwen python=3.10 -y
conda activate dsqwen
python -m pip install --upgrade pip
```

Install PyTorch according to your CUDA version from the official PyTorch selector.

CPU-only example:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

CUDA 12.1 example:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then install project dependencies:

```powershell
pip install -r requirements.txt
```

## 2. Download Qwen3-8B

The base model is required for both evaluation and LoRA training.

**Hugging Face (international):**

```powershell
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-8B --local-dir D:\models\Qwen3-8B
```

**ModelScope (China, faster):**

```powershell
pip install modelscope
modelscope download --model Qwen/Qwen3-8B --local_dir D:\models\Qwen3-8B
```

**Git clone (if CLI tools are unavailable):**

```powershell
# Hugging Face
git clone https://huggingface.co/Qwen/Qwen3-8B D:\models\Qwen3-8B

# Or HF mirror (China)
git clone https://hf-mirror.com/Qwen/Qwen3-8B D:\models\Qwen3-8B
```

> Model page: [Hugging Face](https://huggingface.co/Qwen/Qwen3-8B) | [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-8B)

## 3. Configure environment

```powershell
Copy-Item .env.example .env
notepad .env
```

Set:

```text
DEEPSEEK_API_KEY=your_key
QWEN3_MODEL_PATH=D:\models\Qwen3-8B
DATA_DIR=data_deepseek_hard
HF_LOCAL_FILES_ONLY=1
```

For current PowerShell session:

```powershell
$env:DEEPSEEK_API_KEY="your_key"
$env:QWEN3_MODEL_PATH="D:\models\Qwen3-8B"
$env:DATA_DIR="data_deepseek_hard"
$env:HF_LOCAL_FILES_ONLY="1"
```

## 4. Check environment

```powershell
python scripts/check_env.py
```

## 5. Generate tiny smoke-test dataset

```powershell
python scripts/generate_deepseek_hard_data.py `
  --output_dir data_deepseek_hard `
  --model deepseek-v4-flash `
  --train_per_bucket 2 `
  --test_per_bucket 1 `
  --request_batch_size 5 `
  --sleep 0.5
```

## 6. Validate dataset

```powershell
python scripts/validate_dataset.py --data_dir data_deepseek_hard
```

## 7. Tokenize

```powershell
python scripts/tokenize_deepseek_data.py `
  --model_path $env:QWEN3_MODEL_PATH `
  --data_dir data_deepseek_hard `
  --max_length 256
```

## 8. Evaluate base model

GPU:

```powershell
python scripts/evaluate.py `
  --model_path $env:QWEN3_MODEL_PATH `
  --base_tokenizer_path $env:QWEN3_MODEL_PATH `
  --test_file data_deepseek_hard/test.json `
  --output_dir results/deepseek_hard/base_qwen3 `
  --device cuda:0 `
  --batch_size 2
```

CPU, very slow:

```powershell
python scripts/evaluate.py `
  --model_path $env:QWEN3_MODEL_PATH `
  --base_tokenizer_path $env:QWEN3_MODEL_PATH `
  --test_file data_deepseek_hard/test.json `
  --output_dir results/deepseek_hard/base_qwen3_cpu `
  --device cpu `
  --batch_size 1
```

## 9. Train LoRA

GPU strongly recommended:

```powershell
python scripts/train_lora.py `
  --model_path $env:QWEN3_MODEL_PATH `
  --train_dataset data_deepseek_hard/tokenized_train `
  --eval_dataset data_deepseek_hard/tokenized_test `
  --output_dir ./models/qwen3-8b-lora-deepseek-hard-step300 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 4 `
  --max_steps 300
```

## 10. Evaluate LoRA

```powershell
python scripts/evaluate_lora.py `
  --base_model_path $env:QWEN3_MODEL_PATH `
  --lora_path ./models/qwen3-8b-lora-deepseek-hard-step300 `
  --test_file data_deepseek_hard/test.json `
  --output_dir results/deepseek_hard/lora_step300 `
  --device cuda:0 `
  --batch_size 2
```