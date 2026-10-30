#!/bin/bash
set -euo pipefail

REPOSITORY=/p/project1/cslts/miaari1/python_scripts/global_TL/spatio-temporal-LSTM
STATISTICS_ROOT="${REPOSITORY}/outputs/EU_74_-48_69_20/statisticsEUmap"

if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 <run_name>" >&2
    exit 2
fi

RUN_NAME="$1"
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid run name: use only letters, numbers, '.', '_' or '-'" >&2
    exit 2
fi

RUN_DIR="${STATISTICS_ROOT}/${RUN_NAME}"
if [[ ! -d "${RUN_DIR}/logs" ]]; then
    echo "Run directory is missing: ${RUN_DIR}" >&2
    exit 1
fi

PREDICTION_JOBS_FILE="${RUN_DIR}/submission_state/prediction_jobs.tsv"
if [[ ! -f "$PREDICTION_JOBS_FILE" ]]; then
    echo "Prediction job submissions are not recorded for ${RUN_NAME}: ${PREDICTION_JOBS_FILE}" >&2
    exit 1
fi

declare -A MEMBER_SEEN=()
PREDICTION_JOB_IDS=()
while IFS=$'\t' read -r MEMBER_INDEX JOB_ID; do
    if [[ ! "$MEMBER_INDEX" =~ ^[0-9]+$ ]] ||
       (( MEMBER_INDEX < 0 || MEMBER_INDEX > 99 )) ||
       [[ ! "$JOB_ID" =~ ^[0-9]+$ ]] ||
       [[ -n "${MEMBER_SEEN[$MEMBER_INDEX]:-}" ]]; then
        echo "Invalid or duplicate member/job entry in ${PREDICTION_JOBS_FILE}" >&2
        exit 1
    fi
    MEMBER_SEEN["$MEMBER_INDEX"]=1
    PREDICTION_JOB_IDS+=("$JOB_ID")
done < "$PREDICTION_JOBS_FILE"

if (( ${#PREDICTION_JOB_IDS[@]} != 100 )); then
    echo "Expected 100 prediction jobs, found ${#PREDICTION_JOB_IDS[@]} in ${PREDICTION_JOBS_FILE}" >&2
    exit 1
fi

PREDICTION_DEPENDENCY="afterok"
for MEMBER_INDEX in {0..99}; do
    if [[ -z "${MEMBER_SEEN[$MEMBER_INDEX]:-}" ]]; then
        echo "Prediction job for member ${MEMBER_INDEX} is missing from ${PREDICTION_JOBS_FILE}" >&2
        exit 1
    fi
done
for JOB_ID in "${PREDICTION_JOB_IDS[@]}"; do
    PREDICTION_DEPENDENCY+=":${JOB_ID}"
done

POSTPROCESS_JOB_ID=$(
    sbatch --parsable \
        --dependency="$PREDICTION_DEPENDENCY" \
        --export="ALL,RUN_NAME=${RUN_NAME}" \
        --output="${RUN_DIR}/logs/postprocess_%j.out" \
        --error="${RUN_DIR}/logs/postprocess_%j.err" \
        "${REPOSITORY}/job_statisticsmap_postprocess.sh"
)
POSTPROCESS_JOB_ID="${POSTPROCESS_JOB_ID%%;*}"

printf 'Postprocessing submitted after all 100 member jobs.\n'
