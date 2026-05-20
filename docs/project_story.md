# Project Story

本项目的核心问题不是让模型学习简单情感分类，而是验证：

> 通用大模型在弱情绪、转折表达和边界样本上的情感判断是否稳定，以及能否通过外部强模型数据蒸馏到本地模型。

## 实验流程

1. 使用 DeepSeek 生成 DeepSeek-hard 数据集。
2. 使用 Base Qwen3-8B 在 Hard Set 上直接评测。
3. 使用 DeepSeek-v4-pro 对 Hard Set 做盲测复判，验证标签标准稳定性。
4. 使用 DeepSeek-hard train 数据对 Qwen3-8B 做 LoRA 微调。
5. 在 DeepSeek-hard test 上评测 Qwen3-LoRA。

## 结论

Base Qwen3-8B 在 DeepSeek-hard 上 Accuracy 为 62.50%，说明其对边界情感判断不稳定。

DeepSeek-v4-pro 复判达到 94.03%，说明该 Hard Set 标签具有较高一致性。

Qwen3-8B 经 LoRA 微调后达到 98.61%，说明外部强模型的情感判断标准可以有效蒸馏到本地模型中。
