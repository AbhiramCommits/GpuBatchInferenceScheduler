#!/usr/bin/env bash
#SBATCH --job-name=gpuinfer-bench
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=00:15:00
#SBATCH --output=slurm-bench-%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#
# One-shot batched GEMM benchmark — the same workload as the Kubernetes Job
# in deploy/k8s/job.yaml: C[b] = A[b] * B[b], M=N=K=1024, batch 8, float32,
# 10 iterations via cublasSgemmStridedBatched.
#
# Runs inside the OCI container via srun (Pyxis/enroot) when available,
# otherwise uses the bare binary from ./build/bench_gemm.

set -euo pipefail

M=${M:-1024}
N=${N:-1024}
K=${K:-1024}
BATCH=${BATCH:-8}
ITERS=${ITERS:-10}
IMPL=${IMPL:-cublas}
IMAGE=${GPUINFER_IMAGE:-gpuinfer/scheduler:latest}
RESULTS_DIR=${RESULTS_DIR:-results}

# Job-array sweep support (deploy/slurm/sweep.sh): task i reads line i of
# $SHAPES_FILE, formatted "MxNxKxBATCH".
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -n "${SHAPES_FILE:-}" ]]; then
  shape=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${SHAPES_FILE}")
  IFS=x read -r M N K BATCH <<< "${shape}"
fi

mkdir -p "${RESULTS_DIR}"

# Load a CUDA module when running bare-metal (ignored inside containers).
module load cuda/12.4 2>/dev/null || true

OUT="${RESULTS_DIR}/benchmark-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}.csv"

echo "shape: M=${M} N=${N} K=${K} BATCH=${BATCH} ITERS=${ITERS} IMPL=${IMPL}"

if srun --help 2>&1 | grep -q -- '--container-image'; then
  # Pyxis/enroot is installed: run the benchmark inside the container.
  srun --container-image="${IMAGE}" \
    bench_gemm --m "${M}" --n "${N}" --k "${K}" --batch "${BATCH}" \
      --iters "${ITERS}" --impl "${IMPL}" | tee "${OUT}"
elif [[ -x ./build/bench_gemm ]]; then
  # No container support: run the repo-built binary directly.
  srun ./build/bench_gemm --m "${M}" --n "${N}" --k "${K}" --batch "${BATCH}" \
    --iters "${ITERS}" --impl "${IMPL}" | tee "${OUT}"
else
  echo "error: neither srun container support (Pyxis/enroot) nor" \
    "./build/bench_gemm was found; build the repo first or install Pyxis" >&2
  exit 1
fi
