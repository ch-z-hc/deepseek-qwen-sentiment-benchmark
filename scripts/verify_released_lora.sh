#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# verify_released_lora.sh — One-click verification of the released LoRA adapter
#
# Usage:
#   export QWEN3_MODEL_PATH="/path/to/Qwen3-8B"
#   bash scripts/verify_released_lora.sh
#
# Optional overrides:
#   DATA_DIR=data_deepseek_hard          (default)
#   LORA_PATH=./models/qwen3-8b-lora-deepseek-hard-step300
#   DEVICE=cuda:0
#   BATCH_SIZE=4
# ============================================================================

MODEL_PATH="${QWEN3_MODEL_PATH:?Set QWEN3_MODEL_PATH to your Qwen3-8B directory}"
DATA_DIR="${DATA_DIR:-data_deepseek_hard}"
LORA_PATH="${LORA_PATH:-./models/qwen3-8b-lora-deepseek-hard-step300}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-4}"

echo "============================================"
echo " LoRA Verification Script"
echo "============================================"
echo "MODEL_PATH:  $MODEL_PATH"
echo "DATA_DIR:    $DATA_DIR"
echo "LORA_PATH:   $LORA_PATH"
echo "DEVICE:      $DEVICE"
echo "BATCH_SIZE:  $BATCH_SIZE"
echo "============================================"

# -------------------------------------------------------------------
# [1/5] Check CUDA
# -------------------------------------------------------------------
echo ""
echo "[1/5] Checking CUDA..."

python - <<EOF
import torch
print("torch version :", torch.__version__)
print("torch cuda    :", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu           :", torch.cuda.get_device_name(0))
    print("gpu count     :", torch.cuda.device_count())
EOF

if [ "${DEVICE:0:4}" = "cuda" ]; then
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available but DEVICE=$DEVICE'"
    echo "[OK] CUDA check passed"
fi

# -------------------------------------------------------------------
# [2/5] Validate dataset
# -------------------------------------------------------------------
echo ""
echo "[2/5] Validating dataset..."

if [ ! -f "$DATA_DIR/test.json" ]; then
    echo "ERROR: $DATA_DIR/test.json not found."
    echo "Download the official dataset from:"
    echo "  https://github.com/ch-z-hc/deepseek-qwen-sentiment-benchmark/releases/tag/v1.0"
    echo "  (deepseek_hard_dataset_json.tar.gz)"
    exit 1
fi

python scripts/validate_dataset.py --data_dir "$DATA_DIR"
echo "[OK] Dataset validated"

# -------------------------------------------------------------------
# [3/5] Tokenize
# -------------------------------------------------------------------
echo ""
echo "[3/5] Tokenizing..."

python scripts/tokenize_deepseek_data.py \
  --model_path "$MODEL_PATH" \
  --data_dir "$DATA_DIR" \
  --max_length 256

echo "[OK] Tokenization complete"

# -------------------------------------------------------------------
# [4/5] Evaluate Base Qwen3-8B
# -------------------------------------------------------------------
echo ""
echo "[4/5] Evaluating Base Qwen3-8B..."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/evaluate.py \
  --model_path "$MODEL_PATH" \
  --base_tokenizer_path "$MODEL_PATH" \
  --test_file "$DATA_DIR/test.json" \
  --output_dir results/deepseek_hard/base_qwen3_official_verify \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE"

echo "[OK] Base evaluation complete"

# -------------------------------------------------------------------
# [5/5] Evaluate LoRA
# -------------------------------------------------------------------
echo ""
echo "[5/5] Evaluating Qwen3-8B + LoRA..."

# Check LoRA path exists, handle nested models/models/ case
_actual_lora_path="$LORA_PATH"
if [ ! -d "$_actual_lora_path" ]; then
    _nested="models/$LORA_PATH"
    if [ -d "$_nested" ]; then
        echo "[WARN] LoRA not at $LORA_PATH, but found at $_nested"
        _actual_lora_path="$_nested"
    fi
fi

if [ ! -f "$_actual_lora_path/adapter_config.json" ]; then
    echo "ERROR: LoRA adapter not found at $LORA_PATH"
    echo "Searched locations:"
    find . -name adapter_config.json 2>/dev/null || echo "  (none found)"
    exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/evaluate_lora.py \
  --base_model_path "$MODEL_PATH" \
  --lora_path "$_actual_lora_path" \
  --test_file "$DATA_DIR/test.json" \
  --output_dir results/deepseek_hard/lora_step300_official_verify \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE"

echo "[OK] LoRA evaluation complete"

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
echo ""
echo "============================================"
echo " Verification Complete"
echo "============================================"
echo "Results:"
echo "  Base:   results/deepseek_hard/base_qwen3_official_verify/eval_results.json"
echo "  LoRA:   results/deepseek_hard/lora_step300_official_verify/eval_results.json"
echo ""
echo "Expected (approximate):"
echo "  Base Qwen3-8B accuracy:      ~62%"
echo "  Qwen3-8B + LoRA accuracy:    ~98.6%"
echo "============================================"
