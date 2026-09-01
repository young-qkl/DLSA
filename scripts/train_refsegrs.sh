#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/datasets/RefSegRS}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/dlsa_refsegrs}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT}"

INIT_ARGS=()
if [[ -n "${INIT_CHECKPOINT}" ]]; then
    INIT_ARGS+=(--init_weights "${INIT_CHECKPOINT}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python -m torch.distributed.launch \
    --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" train.py \
    --dataset refsegrs \
    --refsegrs_data_root "${DATA_ROOT}" \
    --model lavt_one \
    --model_id DLSA_RefSegRS \
    --output-dir "${OUTPUT_DIR}" \
    --epochs 20 \
    --lr 3e-5 \
    --batch-size 2 \
    --workers 4 \
    --img_size 480 \
    --swin_type base \
    --window12 \
    --pretrained_swin_weights "${ROOT}/pretrained_weights/swin_base_patch4_window12_384_22k.pth" \
    --ck_bert "${ROOT}/bert-base-uncased" \
    --bert_tokenizer "${ROOT}/bert-base-uncased" \
    --use_text_decoder \
    --text_decoder_variant glf_adapt \
    --text_decoder_stages tg2_tg1 \
    --td_res_scale_init 0.05 \
    --td_gate_bias_init -2.0 \
    --td_alpha_bias_init -1.0 \
    --td_alpha_weight_std 0.001 \
    "${INIT_ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
