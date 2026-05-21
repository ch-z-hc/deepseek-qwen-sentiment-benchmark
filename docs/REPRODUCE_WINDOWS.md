# Windows Reproduction Guide

## 1. Create environment

```powershell
conda create -n dsqwen python=3.10 -y
conda activate dsqwen
python -m pip install --upgrade pip
```

**Step 1:** Install non-PyTorch dependencies first (requirements.txt does NOT include torch):

```powershell
pip install -r requirements.txt
```

**Step 2:** Install PyTorch according to your CUDA version. Check your driver first:

```powershell
nvidia-smi
```

Then install the matching PyTorch:

```powershell
# CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.6:
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### CUDA Compatibility Check

If you see `torch.cuda.is_available() == False` or `The NVIDIA driver on your system is too old`:

```powershell
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```

If CUDA is unavailable, reinstall:

```powershell
pip uninstall -y torch torchvision torchaudio triton
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
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
# For full reproduction (generate data via DeepSeek):
DATA_DIR=data/deepseek_hard
# Or for verifying released LoRA only:
DATA_DIR=data_deepseek_hard
HF_LOCAL_FILES_ONLY=1
```

For current PowerShell session:

```powershell
$env:DEEPSEEK_API_KEY="your_key"
$env:QWEN3_MODEL_PATH="D:\models\Qwen3-8B"
# For full reproduction:
$env:DATA_DIR="data/deepseek_hard"
# Or for verifying released LoRA only:
$env:DATA_DIR="data_deepseek_hard"
$env:HF_LOCAL_FILES_ONLY="1"
```

## 4. Check environment

```powershell
python -m scripts.check_env
```

## 5. Generate tiny smoke-test dataset

```powershell
python -m scripts.generate_deepseek_hard_data `
  --output_dir data/deepseek_hard `
  --model deepseek-v4-flash `
  --train_per_bucket 2 `
  --test_per_bucket 1 `
  --request_batch_size 5 `
  --sleep 0.5
```

## 6. Validate dataset

```powershell
python -m scripts.validate_dataset --data_dir data/deepseek_hard
```

## 7. Tokenize

```powershell
python -m scripts.tokenize_deepseek_data `
  --model_path $env:QWEN3_MODEL_PATH `
  --data_dir data/deepseek_hard `
  --max_length 256
```

## 8. Evaluate base model

GPU:

```powershell
python -m scripts.evaluate `
  --model_path $env:QWEN3_MODEL_PATH `
  --base_tokenizer_path $env:QWEN3_MODEL_PATH `
  --test_file data/deepseek_hard/test.json `
  --output_dir results/deepseek_hard/base_qwen3 `
  --device cuda:0 `
  --batch_size 2
```

CPU, very slow:

```powershell
python -m scripts.evaluate `
  --model_path $env:QWEN3_MODEL_PATH `
  --base_tokenizer_path $env:QWEN3_MODEL_PATH `
  --test_file data/deepseek_hard/test.json `
  --output_dir results/deepseek_hard/base_qwen3_cpu `
  --device cpu `
  --batch_size 1
```

## 9. Train LoRA

GPU strongly recommended:

```powershell
python -m scripts.train_lora `
  --model_path $env:QWEN3_MODEL_PATH `
  --train_dataset data/deepseek_hard/tokenized_train `
  --eval_dataset data/deepseek_hard/tokenized_test `
  --output_dir ./models/qwen3-8b-lora-deepseek-hard-step300 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 4 `
  --max_steps 300
```

## 10. Evaluate LoRA

```powershell
python -m scripts.evaluate_lora `
  --base_model_path $env:QWEN3_MODEL_PATH `
  --lora_path ./models/qwen3-8b-lora-deepseek-hard-step300 `
  --test_file data/deepseek_hard/test.json `
  --output_dir results/deepseek_hard/lora_step300 `
  --device cuda:0 `
  --batch_size 2
```

## Verify Released LoRA (No Data Generation/Training)

If you only want to verify the released LoRA adapter works, download from the [Release page](https://github.com/ch-z-hc/deepseek-qwen-sentiment-benchmark/releases/tag/v1.0):

```powershell
# Download and extract the official dataset
tar -xzf deepseek_hard_dataset_json.tar.gz
# → creates data_deepseek_hard/

# Download and extract the LoRA adapter
# Check internal structure first:
tar -tzf qwen3-8b-lora-deepseek-hard-step300.tar.gz
# If it has models/ prefix, extract to project root:
tar -xzf qwen3-8b-lora-deepseek-hard-step300.tar.gz
# If not, extract into ./models:
mkdir models -Force
tar -xzf qwen3-8b-lora-deepseek-hard-step300.tar.gz -C ./models
```

Set environment:

```powershell
$env:QWEN3_MODEL_PATH="D:\models\Qwen3-8B"
$env:DATA_DIR="data_deepseek_hard"
```

Then run:

```powershell
# Validate dataset
python -m scripts.validate_dataset --data_dir data_deepseek_hard

# Tokenize
python -m scripts.tokenize_deepseek_data `
  --model_path $env:QWEN3_MODEL_PATH `
  --data_dir data_deepseek_hard `
  --max_length 256

# Evaluate Base Qwen3-8B
$env:CUDA_VISIBLE_DEVICES="0"
python -m scripts.evaluate `
  --model_path $env:QWEN3_MODEL_PATH `
  --base_tokenizer_path $env:QWEN3_MODEL_PATH `
  --test_file data_deepseek_hard/test.json `
  --output_dir results/deepseek_hard/base_qwen3_official_verify `
  --device cuda:0 `
  --batch_size 4

# Evaluate LoRA
python -m scripts.evaluate_lora `
  --base_model_path $env:QWEN3_MODEL_PATH `
  --lora_path ./models/qwen3-8b-lora-deepseek-hard-step300 `
  --test_file data_deepseek_hard/test.json `
  --output_dir results/deepseek_hard/lora_step300_official_verify `
  --device cuda:0 `
  --batch_size 4
```

**Expected results:**
- Base Qwen3-8B: ~62% accuracy
- Qwen3-8B + LoRA: ~98.6% accuracy, invalid_count = 0