#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/datasets/RRSIS-D}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/checkpoints/dlsa_rrsisd.pth}"
SPLIT="${SPLIT:-test}"
GPU="${GPU:-0}"

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" python test.py \
    --dataset rrsisd \
    --refer_data_root "${DATA_ROOT}" \
    --resume "${CHECKPOINT}" \
    --split "${SPLIT}" \
    --device cuda:0 \
    --workers 4 \
    --img_size 480 \
    --swin_type base \
    --window12 \
    --ck_bert "${ROOT}/bert-base-uncased" \
    --bert_tokenizer "${ROOT}/bert-base-uncased"
