#!/usr/bin/env bash
# Submits a Slurm job array sweeping the GEMM shape list below (one task per
# shape, each running the same bench_gemm workload as the Kubernetes Job),
# waits for completion, and collates every per-task CSV into one results file.
set -euo pipefail

SHAPES=(
  "512x512x512x2"
  "1024x1024x1024x8"
  "2048x2048x2048x4"
  "4096x4096x4096x2"
)
IMPL=${IMPL:-cublas}
RESULTS_DIR=${RESULTS_DIR:-results}

mkdir -p "${RESULTS_DIR}"
SHAPES_FILE="${RESULTS_DIR}/shapes.txt"
printf '%s\n' "${SHAPES[@]}" > "${SHAPES_FILE}"

JOBID=$(sbatch --array=1-${#SHAPES[@]} \
  --export=SHAPES_FILE="${SHAPES_FILE}",RESULTS_DIR="${RESULTS_DIR}",IMPL="${IMPL}" \
  deploy/slurm/submit.sbatch | awk '{print $NF}')
echo "submitted array job ${JOBID} over ${#SHAPES[@]} shapes"

echo "waiting for array job ${JOBID} to finish..."
while squeue -j "${JOBID}" -h -o '%T' 2>/dev/null | grep -q .; do
  sleep 30
done

OUT="${RESULTS_DIR}/sweep.csv"
first=1
for f in "${RESULTS_DIR}"/benchmark-*.csv; do
  [[ -e "$f" ]] || continue
  if [[ $first -eq 1 ]]; then
    cat "$f" > "${OUT}"
    first=0
  else
    tail -n +2 "$f" >> "${OUT}"
  fi
done

echo "collated ${#SHAPES[@]} shape results into ${OUT}:"
cat "${OUT}"
