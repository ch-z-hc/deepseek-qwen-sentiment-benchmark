# DeepSeek Teacher to Qwen3 Student: 涓枃鎯呮劅鍒嗘瀽 Hard Set 涓?LoRA 钂搁寰皟

鏈」鐩娇鐢?DeepSeek 浣滀负澶栭儴寮烘ā鍨?Teacher锛屾瀯寤哄棰嗗煙涓枃鎯呮劅鍒嗘瀽 Hard Set锛屽苟浣跨敤鏈湴 Qwen3-8B 浣滀负 Student 妯″瀷杩涜 PEFT/LoRA 鍙傛暟楂樻晥寰皟銆?
椤圭洰閲嶇偣涓嶆槸绠€鍗曟儏鎰熷垎绫伙紝鑰屾槸楠岃瘉鏈湴澶фā鍨嬪湪寮辨儏缁€佽浆鎶樿〃杈俱€佽竟鐣屾牱鏈笂鐨勭ǔ瀹氭€э紝骞堕€氳繃 DeepSeek Teacher 鏁版嵁钂搁鎻愬崌鏈湴 Qwen3-8B 鐨勫垽鏂兘鍔涖€?
## Final Results

DeepSeek-hard 娴嬭瘯闆嗭紝鍏?1440 鏉℃牱鏈€?
| Model / Annotator | Test Set | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Base Qwen3-8B | DeepSeek-hard | 62.50% | 59.48% |
| DeepSeek-v4-pro | DeepSeek-hard | 94.03% | 93.89% |
| Qwen3-8B LoRA step300 | DeepSeek-hard | **98.61%** | **98.61%** |

## Project Highlights

- 浣跨敤 DeepSeek 鏋勫缓瑕嗙洊 8 涓鍩熴€? 绫绘儏鎰熺殑涓枃鐭瘎璁?Hard Set銆?- Hard Set 鑱氱劍寮辨儏缁€佽浆鎶樿〃杈惧拰杈圭晫鏍锋湰銆?- 浣跨敤 DeepSeek-v4-pro 杩涜澶栭儴澶嶅垽锛屽鍒ゅ噯纭巼杈?94.03%銆?- 浣跨敤 PEFT/LoRA 瀵规湰鍦?Qwen3-8B 杩涜鍙傛暟楂樻晥寰皟銆?- LoRA 寰皟鍚庯紝Hard Set Accuracy 浠?62.50% 鎻愬崌鍒?98.61%銆?
## Repository Structure

    .
    鈹溾攢鈹€ README.md
    鈹溾攢鈹€ requirements.txt
    鈹溾攢鈹€ .env.example
    鈹溾攢鈹€ .gitignore
    鈹溾攢鈹€ configs/
    鈹溾攢鈹€ docs/
    鈹溾攢鈹€ scripts/
    鈹溾攢鈹€ data/
    鈹?  鈹溾攢鈹€ deepseek_hard/
    鈹?  鈹斺攢鈹€ samples/
    鈹溾攢鈹€ results/
    鈹?  鈹溾攢鈹€ deepseek_hard/
    鈹?  鈹斺攢鈹€ final_summary/
    鈹斺攢鈹€ models/
        鈹斺攢鈹€ README_MODEL_ARTIFACT.md

## Environment

    conda activate llm-eval
    pip install -r requirements.txt
    export DEEPSEEK_API_KEY="your_key"

鏈」鐩粯璁ゆ湰鍦版ā鍨嬭矾寰勶細

    /data/lys/models/Qwen3-8B

## Reproduce

### 1. Generate DeepSeek Hard Set

    python scripts/generate_deepseek_hard_data.py \
      --output_dir data/deepseek_hard \
      --model deepseek-v4-flash \
      --train_per_bucket 200 \
      --test_per_bucket 60

### 2. Tokenize

    python scripts/tokenize_deepseek_data.py \
      --model_path /data/lys/models/Qwen3-8B \
      --data_dir data/deepseek_hard

### 3. Evaluate Base Qwen3-8B

    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
      --model_path /data/lys/models/Qwen3-8B \
      --test_file data/deepseek_hard/test.json \
      --output_dir results/deepseek_hard/base_qwen3 \
      --device cuda:0 \
      --batch_size 4

### 4. DeepSeek-v4-pro External Recheck

    python scripts/evaluate_deepseek_classifier.py \
      --test_file data/deepseek_hard/test.json \
      --output_dir results/deepseek_hard/deepseek_classifier_pro \
      --model deepseek-v4-pro \
      --batch_size 20

### 5. Train Qwen3-LoRA

    CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py \
      --model_path /data/lys/models/Qwen3-8B \
      --train_dataset data/deepseek_hard/tokenized_train \
      --eval_dataset data/deepseek_hard/tokenized_test \
      --output_dir ./models/qwen3-8b-lora-deepseek-hard-step300 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 4 \
      --max_steps 300

### 6. Evaluate Qwen3-LoRA

    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_lora.py \
      --base_model_path /data/lys/models/Qwen3-8B \
      --lora_path ./models/qwen3-8b-lora-deepseek-hard-step300 \
      --test_file data/deepseek_hard/test.json \
      --output_dir results/deepseek_hard/lora_step300 \
      --device cuda:0 \
      --batch_size 4

## Model Artifact

The final LoRA adapter is provided as a GitHub Release asset:

    qwen3-8b-lora-deepseek-hard-step300.tar.gz

The base Qwen3-8B model is not included.

## Resume Version

    \item 浣跨敤 DeepSeek 浣滀负澶栭儴寮烘ā鍨嬫瀯寤鸿鐩?8 涓鍩熴€? 绫绘儏鎰熺殑涓枃鐭瘎璁?Hard Set锛岄噸鐐瑰寘鍚急鎯呯华銆佽浆鎶樿〃杈惧拰杈圭晫鏍锋湰
    \item 鍩轰簬鏈湴 Qwen3-8B 鏋勫缓瀛︾敓妯″瀷锛屼娇鐢?PEFT/LoRA 杩涜鍙傛暟楂樻晥寰皟锛屽皢 DeepSeek 鐨勬儏鎰熸爣娉ㄦ爣鍑嗚捀棣忓埌鏈湴妯″瀷
    \item 鍦?1440 鏉?DeepSeek-hard 娴嬭瘯闆嗕笂锛孊ase Qwen3-8B Accuracy 涓?62.50\%锛孡oRA 寰皟鍚庢彁鍗囪嚦 98.61\%锛孧acro-F1 浠?59.48\% 鎻愬崌鑷?98.61\%
    \item 寮曞叆 DeepSeek-v4-pro 杩涜澶栭儴澶嶅垽锛屽鍒ゅ噯纭巼杈?94.03\%锛岄獙璇?Hard Set 鏍囩鏍囧噯鍏锋湁杈冮珮涓€鑷存€?
