#!/usr/bin/env bash
# Profile a workload with Nsight Systems (timeline + stats); optionally run
# Nsight Compute on the tiled GEMM.
#
# Requirements: a CUDA GPU, nsys on PATH, and the repo built
# (cmake -B build && cmake --build build -j).
#
# Environment overrides:
#   WORKLOAD   gemm | scheduler        (default: gemm)
#   OUTDIR     output directory        (default: results/profiling)
#   TRACE      nsys trace domains      (default: cuda,nvtx,osrt)
#   NCU        1 to run ncu, 0 to skip (default: 1)
#   BENCH_BIN  GEMM benchmark binary   (default: ./build/bench_gemm)
#   SCHED_BIN  scheduler daemon binary (default: ./build/scheduler_daemon)
#   M N K BATCH ITERS IMPL WARMUP      GEMM parameters
#   SKIP_CHECK 1 to skip the CPU reference during GEMM profiling
#   JOBS ARRIVAL_RATE WORKERS          scheduler parameters
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTDIR=${OUTDIR:-"${REPO_ROOT}/results/profiling"}
WORKLOAD=${WORKLOAD:-gemm}
TRACE=${TRACE:-cuda,nvtx,osrt}
NCU=${NCU:-1}

BENCH_BIN=${BENCH_BIN:-"${REPO_ROOT}/build/bench_gemm"}
SCHED_BIN=${SCHED_BIN:-"${REPO_ROOT}/build/scheduler_daemon"}

M=${M:-1024}
N=${N:-1024}
K=${K:-1024}
BATCH=${BATCH:-8}
ITERS=${ITERS:-10}
WARMUP=${WARMUP:-10}
IMPL=${IMPL:-tiled}
SKIP_CHECK=${SKIP_CHECK:-0}

JOBS=${JOBS:-1000}
ARRIVAL_RATE=${ARRIVAL_RATE:-20}
WORKERS=${WORKERS:-2}

mkdir -p "${OUTDIR}"

if ! command -v nsys >/dev/null 2>&1; then
  echo "error: nsys not on PATH (install Nsight Systems / CUDA toolkit)" >&2
  exit 1
fi

NSYS_CMD=(
  nsys profile
  --trace="${TRACE}"
  --stats=true
  --force-overwrite=true
  -o "${OUTDIR}/nsys_report"
)

if [[ "${WORKLOAD}" == "scheduler" ]]; then
  if [[ ! -x "${SCHED_BIN}" ]]; then
    echo "error: ${SCHED_BIN} not found; run: cmake -B build && cmake --build build -j" >&2
    exit 1
  fi
  echo "nsys: profiling scheduler (jobs=${JOBS} arrival_rate=${ARRIVAL_RATE} workers=${WORKERS})"
  "${NSYS_CMD[@]}" "${SCHED_BIN}" \
    --jobs "${JOBS}" --arrival-rate "${ARRIVAL_RATE}" --workers "${WORKERS}" \
    --http-port 0 --metrics-csv "${OUTDIR}/scheduler_metrics.csv" \
    2>&1 | tee "${OUTDIR}/nsys_stats.txt"
else
  if [[ ! -x "${BENCH_BIN}" ]]; then
    echo "error: ${BENCH_BIN} not found; run: cmake -B build && cmake --build build -j" >&2
    exit 1
  fi
  echo "nsys: profiling ${BENCH_BIN} (M=${M} N=${N} K=${K} batch=${BATCH} iters=${ITERS} warmup=${WARMUP} impl=${IMPL})"
  GEMM_EXTRA=()
  [[ "${SKIP_CHECK}" == "1" ]] && GEMM_EXTRA+=(--skip-check)
  "${NSYS_CMD[@]}" "${BENCH_BIN}" \
    --m "${M}" --n "${N}" --k "${K}" --batch "${BATCH}" \
    --iters "${ITERS}" --warmup "${WARMUP}" --impl "${IMPL}" "${GEMM_EXTRA[@]}" \
    2>&1 | tee "${OUTDIR}/nsys_stats.txt"

  if [[ "${NCU}" != "0" ]] && command -v ncu >/dev/null 2>&1; then
    echo "ncu: profiling the tiled kernel"
    ncu --set full --force-overwrite \
      --export "${OUTDIR}/ncu_tiled" \
      "${BENCH_BIN}" --m "${M}" --n "${N}" --k "${K}" --batch "${BATCH}" \
        --iters 1 --warmup 0 --impl tiled --skip-check
  elif [[ "${NCU}" != "0" ]]; then
    echo "warning: ncu not found; skipping kernel-level profiling" >&2
  fi
fi

if [[ ! -f "${OUTDIR}/nsys_report.sqlite" ]]; then
  echo "exporting sqlite statistics"
  nsys stats --format sqlite --output "${OUTDIR}/nsys_report" \
    "${OUTDIR}/nsys_report.nsys-rep" >/dev/null 2>&1 || true
fi

if [[ -f "${OUTDIR}/nsys_report.sqlite" ]]; then
  echo "parsing nsys report"
  python3 "${SCRIPT_DIR}/parse_nsys.py" "${OUTDIR}/nsys_report.sqlite" \
    -o "${OUTDIR}/nsys_breakdown.md"
else
  echo "warning: ${OUTDIR}/nsys_report.sqlite not found; skipping breakdown" >&2
fi

echo "profiling artifacts in ${OUTDIR}:"
for f in "${OUTDIR}"/nsys_report* "${OUTDIR}"/ncu_tiled* "${OUTDIR}"/nsys_breakdown.md \
         "${OUTDIR}"/nsys_stats.txt; do
  [[ -e "$f" ]] && printf '%s\n' "$f"
done
