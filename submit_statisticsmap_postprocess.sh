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
declare -A MEMBER_JOB_IDS=()
while IFS=$'\t' read -r MEMBER_INDEX JOB_ID; do
    if [[ ! "$MEMBER_INDEX" =~ ^[0-9]+$ ]] ||
       (( MEMBER_INDEX < 0 || MEMBER_INDEX > 99 )) ||
       [[ ! "$JOB_ID" =~ ^[0-9]+$ ]] ||
       [[ -n "${MEMBER_SEEN[$MEMBER_INDEX]:-}" ]]; then
        echo "Invalid or duplicate member/job entry in ${PREDICTION_JOBS_FILE}" >&2
        exit 1
    fi
    MEMBER_SEEN["$MEMBER_INDEX"]=1
    MEMBER_JOB_IDS["$MEMBER_INDEX"]="$JOB_ID"
done < "$PREDICTION_JOBS_FILE"

if (( ${#MEMBER_JOB_IDS[@]} != 100 )); then
    echo "Expected 100 prediction jobs, found ${#MEMBER_JOB_IDS[@]} in ${PREDICTION_JOBS_FILE}" >&2
    exit 1
fi

INCOMPLETE_JOB_IDS=()
for MEMBER_INDEX in {0..99}; do
    if [[ -z "${MEMBER_SEEN[$MEMBER_INDEX]:-}" ]]; then
        echo "Prediction job for member ${MEMBER_INDEX} is missing from ${PREDICTION_JOBS_FILE}" >&2
        exit 1
    fi
    printf -v MEMBER_LABEL '%03d' "$MEMBER_INDEX"
    if [[ ! -s "${RUN_DIR}/predictions/daily_wtd_member_${MEMBER_LABEL}.npy" ||
          ! -s "${RUN_DIR}/metadata/member_${MEMBER_LABEL}_prediction.json" ]]; then
        INCOMPLETE_JOB_IDS+=("${MEMBER_JOB_IDS[$MEMBER_INDEX]}")
    fi
done

SBATCH_ARGS=(
    --parsable
    --export="ALL,RUN_NAME=${RUN_NAME}"
    --output="${RUN_DIR}/logs/postprocess_%j.out"
    --error="${RUN_DIR}/logs/postprocess_%j.err"
)

if (( ${#INCOMPLETE_JOB_IDS[@]} > 0 )); then
    JOB_LIST=$(IFS=,; printf '%s' "${INCOMPLETE_JOB_IDS[*]}")
    ACTIVE_JOB_IDS=$(squeue --noheader --jobs="$JOB_LIST" --format='%i')
    declare -A ACTIVE_JOB_SEEN=()
    while IFS= read -r ACTIVE_JOB_ID; do
        [[ -n "$ACTIVE_JOB_ID" ]] && ACTIVE_JOB_SEEN["$ACTIVE_JOB_ID"]=1
    done <<< "$ACTIVE_JOB_IDS"

    PREDICTION_DEPENDENCY="afterok"
    for JOB_ID in "${INCOMPLETE_JOB_IDS[@]}"; do
        if [[ -z "${ACTIVE_JOB_SEEN[$JOB_ID]:-}" ]]; then
            echo "Prediction output is incomplete for member job ${JOB_ID}, but that job is no longer queued or running." >&2
            echo "Check the member logs and resubmit or repair the incomplete member before postprocessing." >&2
            exit 1
        fi
        PREDICTION_DEPENDENCY+=":${JOB_ID}"
    done
    SBATCH_ARGS+=(--dependency="$PREDICTION_DEPENDENCY")
else
    echo "All 100 member outputs and completion manifests are present; submitting postprocessing without expired Slurm dependencies."
fi

POSTPROCESS_JOB_ID=$(
    sbatch "${SBATCH_ARGS[@]}" \
        "${REPOSITORY}/job_statisticsmap_postprocess.sh"
)
POSTPROCESS_JOB_ID="${POSTPROCESS_JOB_ID%%;*}"

printf 'Postprocessing submitted as job %s.\n' "$POSTPROCESS_JOB_ID"
