#!/usr/bin/env bash
set -euo pipefail

python scripts/train_timm_backbone.py \
  --model-name convnextv2_tiny \
  --display-name "ConvNeXt-V2 Tiny LF+DSCL" \
  --data-dir data \
  --output-dir models/convnextv2_lf_dscl \
  --checkpoint-name best_convnextv2_tiny_lf_dscl.pth \
  --summary-name convnextv2_tiny_lf_dscl_summary.json \
  --epochs 100 \
  --batch-size 16 \
  --learning-rate 0.0001 \
  --lambda-scl 0.1 \
  --patience 5 \
  --use-scheduler \
  --scheduler-patience 3 \
  --scheduler-factor 0.1 \
  --seed 42 \
  --test-size 0.2 \
  --val-size 0.1 \
  --image-size 64 \
  --use-lf \
  --use-dscl \
  --contrastive-loss dscl 2>&1 | tee models/convnextv2_lf_dscl/training_log.txt
