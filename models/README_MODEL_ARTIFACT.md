# Model Artifact

The final LoRA adapter is provided as a GitHub Release asset:

    qwen3-8b-lora-deepseek-hard-step300.tar.gz

After downloading:

    tar -xzf qwen3-8b-lora-deepseek-hard-step300.tar.gz

Then evaluate with:

    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_lora.py ^
      --base_model_path ./models/Qwen3-8B \
      --lora_path ./models/qwen3-8b-lora-deepseek-hard-step300 \
      --test_file data/deepseek_hard/test.json \
      --output_dir results/deepseek_hard/lora_step300 \
      --device cuda:0 \
      --batch_size 4

The base Qwen3-8B model is not included in this repository.
