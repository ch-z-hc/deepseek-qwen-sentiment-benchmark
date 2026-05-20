# Experiment Results

## DeepSeek-hard

测试集规模：1440 条  
领域数量：8  
情感类别：负面 / 中性 / 正面  
每类样本数：480

| Model / Annotator | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Base Qwen3-8B | 62.50% | 59.48% | 59.48% |
| DeepSeek-v4-pro | 94.03% | 93.89% | 93.89% |
| Qwen3-8B LoRA step300 | 98.61% | 98.61% | 98.61% |

## Key Finding

Base Qwen3-8B 在明确情感样本上表现较强，但在 Hard Set 中的弱情绪、转折表达、边界样本上明显不稳定。

LoRA 微调后，本地 Qwen3-8B 显著对齐 DeepSeek Teacher 的边界情感判断标准。
