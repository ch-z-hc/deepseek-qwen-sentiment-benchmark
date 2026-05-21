# Model Artifact

The final LoRA adapter is provided as a GitHub Release asset:

    qwen3-8b-lora-deepseek-hard-step300.tar.gz

## Extraction

The tarball may have an internal `models/` prefix. Check first:

```bash
# Preview the tarball structure:
tar -tzf qwen3-8b-lora-deepseek-hard-step300.tar.gz | head -5

# If the listing shows models/... at top level, extract to project root:
tar -xzf qwen3-8b-lora-deepseek-hard-step300.tar.gz

# If there is no models/ prefix, extract into ./models:
mkdir -p models
tar -xzf qwen3-8b-lora-deepseek-hard-step300.tar.gz -C ./models
```

Verify the adapter is at the right path:

```bash
ls models/qwen3-8b-lora-deepseek-hard-step300/adapter_config.json
```

If you end up with `models/models/qwen3-8b-lora-deepseek-hard-step300/`, move it up:

```bash
mv models/models/qwen3-8b-lora-deepseek-hard-step300 models/
```

## Evaluation

Using the official release dataset (`data_deepseek_hard`):

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_lora \
  --base_model_path ./models/Qwen3-8B \
  --lora_path ./models/qwen3-8b-lora-deepseek-hard-step300 \
  --test_file data_deepseek_hard/test.json \
  --output_dir results/deepseek_hard/lora_step300 \
  --device cuda:0 \
  --batch_size 4
```

The base Qwen3-8B model is not included in this repository.
