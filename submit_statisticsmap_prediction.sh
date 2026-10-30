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
JOB_STATE_DIR="${RUN_DIR}/submission_state"
PREPROCESS_JOB_ID_FILE="${JOB_STATE_DIR}/preprocess_job_id"
PREPROCESS_JOB_ID=""
if [[ -f "$PREPROCESS_JOB_ID_FILE" ]]; then
    IFS= read -r PREPROCESS_JOB_ID < "$PREPROCESS_JOB_ID_FILE" || true
    if [[ ! "$PREPROCESS_JOB_ID" =~ ^[0-9]+$ ]]; then
        echo "Stored preprocessing job ID is invalid: ${PREPROCESS_JOB_ID_FILE}" >&2
        exit 1
    fi
fi
if [[ -z "$PREPROCESS_JOB_ID" ]] &&
   [[ ! -s "${RUN_DIR}/feature_inputs/era5land_features.npy" ||
      ! -f "${RUN_DIR}/metadata/preprocessing.json" ]]; then
    echo "Preprocessed inputs are missing and no preprocessing submission is recorded for ${RUN_NAME}" >&2
    exit 1
fi

mkdir -p "$JOB_STATE_DIR"
PREDICTION_JOBS_FILE="${JOB_STATE_DIR}/prediction_jobs.tsv"
if [[ -e "$PREDICTION_JOBS_FILE" ]]; then
    echo "Prediction submissions are already recorded for ${RUN_NAME}: ${PREDICTION_JOBS_FILE}" >&2
    echo "Use a new run name to avoid duplicate member jobs or overwriting outputs." >&2
    exit 1
fi

if ! (set -o noclobber; : > "$PREDICTION_JOBS_FILE"); then
    echo "Could not create prediction submission record: ${PREDICTION_JOBS_FILE}" >&2
    exit 1
fi

for MEMBER_INDEX in {0..99}; do
    printf -v MEMBER_LABEL '%03d' "$MEMBER_INDEX"
    SBATCH_ARGS=(
        --parsable
        --job-name="statsmap_member_${MEMBER_LABEL}"
        --export="ALL,RUN_NAME=${RUN_NAME},MEMBER_INDEX=${MEMBER_INDEX}"
        --output="${RUN_DIR}/logs/member_${MEMBER_LABEL}_%j.out"
        --error="${RUN_DIR}/logs/member_${MEMBER_LABEL}_%j.err"
    )
    if [[ -n "$PREPROCESS_JOB_ID" ]]; then
        SBATCH_ARGS+=(--dependency="afterok:${PREPROCESS_JOB_ID}")
    fi

    JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" \
        "${REPOSITORY}/job_statisticsmap_era5land_array.sh")
    JOB_ID="${JOB_ID%%;*}"
    if [[ ! "$JOB_ID" =~ ^[0-9]+$ ]]; then
        echo "Slurm returned an invalid job ID for member ${MEMBER_INDEX}: ${JOB_ID}" >&2
        exit 1
    fi
    printf '%s\t%s\n' "$MEMBER_INDEX" "$JOB_ID" >> "$PREDICTION_JOBS_FILE"
    printf 'Submitted member %s as job %s\n' "$MEMBER_INDEX" "$JOB_ID"
done

printf 'Submitted 100 independent prediction jobs for members 0-99.\n'
printf 'Member-to-job mapping saved in %s\n' "$PREDICTION_JOBS_FILE"
