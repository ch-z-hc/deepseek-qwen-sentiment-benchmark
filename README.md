# DeepSeek Teacher to Qwen3 Student: 中文情感分析 Hard Set 与 LoRA 蒸馏微调

本项目使用 DeepSeek 作为外部强模型 Teacher，构建多领域中文情感分析 Hard Set，并使用本地 Qwen3-8B 作为 Student 模型进行 PEFT/LoRA 参数高效微调。

项目重点不是简单情感分类，而是验证本地大模型在弱情绪、转折表达、边界样本上的稳定性，并通过 DeepSeek Teacher 数据蒸馏提升本地 Qwen3-8B 的判断能力。

## Final Results

DeepSeek-hard 测试集，共 1440 条样本。

| Model / Annotator | Test Set | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Base Qwen3-8B | DeepSeek-hard | 62.50% | 59.48% |
| DeepSeek-v4-pro | DeepSeek-hard | 94.03% | 93.89% |
| Qwen3-8B LoRA step300 | DeepSeek-hard | **98.61%** | **98.61%** |

## Project Highlights

- 使用 DeepSeek 构建覆盖 8 个领域、3 类情感的中文短评论 Hard Set。
- Hard Set 聚焦弱情绪、转折表达和边界样本。
- 使用 DeepSeek-v4-pro 进行外部复判，复判准确率达 94.03%。
- 使用 PEFT/LoRA 对本地 Qwen3-8B 进行参数高效微调。
- LoRA 微调后，Hard Set Accuracy 从 62.50% 提升到 98.61%。

## Repository Structure

    .
    ├── README.md
    ├── requirements.txt
    ├── .env.example
    ├── .gitignore
    ├── configs/
    ├── docs/
    ├── scripts/
    ├── data/
    │   ├── deepseek_hard/
    │   └── samples/
    ├── results/
    │   ├── deepseek_hard/
    │   └── final_summary/
    └── models/
        └── README_MODEL_ARTIFACT.md

## Environment

    conda activate llm-eval
    pip install -r requirements.txt
    export DEEPSEEK_API_KEY="your_key"

本项目默认本地模型路径：

    /data/lys/models/Qwen3-8B

## Reproduce

### 1. Generate DeepSeek Hard Set

    python scripts/generate_deepseek_hard_data.py \
      --output_dir data_deepseek_hard \
      --model deepseek-v4-flash \
      --train_per_bucket 200 \
      --test_per_bucket 60

### 2. Tokenize

    python scripts/tokenize_deepseek_data.py \
      --model_path /data/lys/models/Qwen3-8B \
      --data_dir data_deepseek_hard

### 3. Evaluate Base Qwen3-8B

    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
      --model_path /data/lys/models/Qwen3-8B \
      --test_file data_deepseek_hard/test.json \
      --output_dir results/deepseek_hard/base_qwen3 \
      --device cuda:0 \
      --batch_size 4

### 4. DeepSeek-v4-pro External Recheck

    python scripts/evaluate_deepseek_classifier.py \
      --test_file data_deepseek_hard/test.json \
      --output_dir results/deepseek_hard/deepseek_classifier_pro \
      --model deepseek-v4-pro \
      --batch_size 20

### 5. Train Qwen3-LoRA

    CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py \
      --model_path /data/lys/models/Qwen3-8B \
      --train_dataset data_deepseek_hard/tokenized_train \
      --eval_dataset data_deepseek_hard/tokenized_test \
      --output_dir ./models/qwen3-8b-lora-deepseek-hard-step300 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 4 \
      --max_steps 300

### 6. Evaluate Qwen3-LoRA

    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_lora.py \
      --base_model_path /data/lys/models/Qwen3-8B \
      --lora_path ./models/qwen3-8b-lora-deepseek-hard-step300 \
      --test_file data_deepseek_hard/test.json \
      --output_dir results/deepseek_hard/lora_step300 \
      --device cuda:0 \
      --batch_size 4

## Model Artifact

The final LoRA adapter is provided as a GitHub Release asset:

    qwen3-8b-lora-deepseek-hard-step300.tar.gz

The base Qwen3-8B model is not included.

## Resume Version

    \item 使用 DeepSeek 作为外部强模型构建覆盖 8 个领域、3 类情感的中文短评论 Hard Set，重点包含弱情绪、转折表达和边界样本
    \item 基于本地 Qwen3-8B 构建学生模型，使用 PEFT/LoRA 进行参数高效微调，将 DeepSeek 的情感标注标准蒸馏到本地模型
    \item 在 1440 条 DeepSeek-hard 测试集上，Base Qwen3-8B Accuracy 为 62.50\%，LoRA 微调后提升至 98.61\%，Macro-F1 从 59.48\% 提升至 98.61\%
    \item 引入 DeepSeek-v4-pro 进行外部复判，复判准确率达 94.03\%，验证 Hard Set 标签标准具有较高一致性
