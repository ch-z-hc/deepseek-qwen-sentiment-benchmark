#!/usr/bin/env bash
# One-click environment setup for CUDA 12.6.
# Usage: bash scripts/setup_env_cu126.sh
set -euo pipefail

echo "==> Installing PyTorch (CUDA 12.6) ..."
pip install -U pip
pip install --no-cache-dir \
  torch==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126

echo "==> Pinning torch to prevent accidental upgrade ..."
cat > constraints-cu126.txt <<'EOF'
torch==2.7.0
EOF

echo "==> Installing project dependencies ..."
pip install -r requirements.txt -c constraints-cu126.txt

echo "==> Verifying CUDA ..."
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available. Your NVIDIA driver may be too old for the CUDA 12.6 wheel.\n"
        "Run 'nvidia-smi' to check your driver version."
    )
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "==> Environment ready."
