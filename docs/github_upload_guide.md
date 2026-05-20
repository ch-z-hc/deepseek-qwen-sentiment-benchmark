# GitHub Upload Guide

## Recommended repository files

Commit the clean repository:

    git add README.md requirements.txt .gitignore .env.example configs docs scripts data results models
    git commit -m "Initial commit: DeepSeek teacher to Qwen3 LoRA sentiment benchmark"

## Release assets

Upload files from release_assets/ to GitHub Release:

- qwen3-8b-lora-deepseek-hard-step300.tar.gz
- deepseek_hard_dataset_json.tar.gz
- final_results_deepseek_hard.tar.gz

## Do not commit

- DeepSeek API Key
- Local Qwen3-8B base model
- .cache/
- wandb/
- tokenized datasets
- intermediate failed checkpoints
