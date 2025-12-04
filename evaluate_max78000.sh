#!/bin/sh
# Evaluation script for Animals

MODEL="ai85cdnet"
DATASET="paper_glass"
QUANTIZED_MODEL="../ai8x-training/logs/2025.11.09-225951/best-quantized.pth.tar"

# Run evaluation
python train.py \
  --arch "$MODEL" \
  --dataset "$DATASET" \
  --confusion \
  --evaluate \
  --exp-load-weights-from "$QUANTIZED_MODEL" \
  --8-bit-mode \
  --save-sample 1 \
  --device MAX78000 "$@"
