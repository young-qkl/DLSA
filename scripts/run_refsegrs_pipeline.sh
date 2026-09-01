#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${STAGE:-all}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/datasets/RefSegRS}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${ROOT}/checkpoints/base_refsegrs}"
DLSA_OUTPUT_DIR="${DLSA_OUTPUT_DIR:-${ROOT}/checkpoints/dlsa_refsegrs}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${BASE_OUTPUT_DIR}/model_best_RMSIN_RefSegRS.pth}"
DLSA_CHECKPOINT="${DLSA_CHECKPOINT:-${DLSA_OUTPUT_DIR}/model_best_DLSA_RefSegRS.pth}"

run_base() {
    DATA_ROOT="${DATA_ROOT}" OUTPUT_DIR="${BASE_OUTPUT_DIR}" \
        bash "${ROOT}/scripts/train_refsegrs_base.sh"
}

run_dlsa() {
    [[ -f "${BASE_CHECKPOINT}" ]] || { echo "Missing base checkpoint: ${BASE_CHECKPOINT}" >&2; exit 1; }
    DATA_ROOT="${DATA_ROOT}" OUTPUT_DIR="${DLSA_OUTPUT_DIR}" \
        INIT_CHECKPOINT="${BASE_CHECKPOINT}" bash "${ROOT}/scripts/train_refsegrs.sh"
}

run_test() {
    [[ -f "${DLSA_CHECKPOINT}" ]] || { echo "Missing DLSA checkpoint: ${DLSA_CHECKPOINT}" >&2; exit 1; }
    mkdir -p "${DLSA_OUTPUT_DIR}"
    for split in val test; do
        DATA_ROOT="${DATA_ROOT}" CHECKPOINT="${DLSA_CHECKPOINT}" SPLIT="${split}" \
            bash "${ROOT}/scripts/test_refsegrs.sh" | tee "${DLSA_OUTPUT_DIR}/${split}.log"
    done
}

case "${STAGE}" in
    all) run_base; run_dlsa; run_test ;;
    base) run_base ;;
    dlsa) run_dlsa ;;
    test) run_test ;;
    *) echo "STAGE must be one of: all, base, dlsa, test" >&2; exit 2 ;;
esac
