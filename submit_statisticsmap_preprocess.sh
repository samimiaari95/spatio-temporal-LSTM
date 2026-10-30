#!/bin/bash
set -euo pipefail

REPOSITORY=/p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM
STATISTICS_ROOT="${REPOSITORY}/outputs/EU_74_-48_69_20/statisticsEUmap"

if [[ "$#" -gt 1 ]]; then
    echo "Usage: $0 [run_name]" >&2
    exit 2
fi

RUN_NAME="${1:-run_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid run name: use only letters, numbers, '.', '_' or '-'" >&2
    exit 2
fi

RUN_DIR="${STATISTICS_ROOT}/${RUN_NAME}"
if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to reuse existing run directory: ${RUN_DIR}" >&2
    exit 1
fi

mkdir -p "${STATISTICS_ROOT}/logs"
mkdir "$RUN_DIR"
mkdir "${RUN_DIR}/logs"

PREPROCESS_JOB_ID=$(
    sbatch --parsable \
        --export="ALL,RUN_NAME=${RUN_NAME}" \
        --output="${RUN_DIR}/logs/preprocess_%j.out" \
        --error="${RUN_DIR}/logs/preprocess_%j.err" \
        "${REPOSITORY}/job_statisticsmap_preprocess.sh"
)
PREPROCESS_JOB_ID="${PREPROCESS_JOB_ID%%;*}"

JOB_STATE_DIR="${RUN_DIR}/submission_state"
mkdir -p "$JOB_STATE_DIR"
TEMP_JOB_ID_FILE="${JOB_STATE_DIR}/.preprocess_job_id.tmp.$$"
printf '%s\n' "$PREPROCESS_JOB_ID" > "$TEMP_JOB_ID_FILE"
mv -f "$TEMP_JOB_ID_FILE" "${JOB_STATE_DIR}/preprocess_job_id"

printf 'RUN_NAME=%s\n' "$RUN_NAME"
printf 'Preprocessing submitted; dependency information saved for this run.\n'
