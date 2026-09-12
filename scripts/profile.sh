#!/usr/bin/env bash
# Profile the GEMM benchmark with Nsight Systems (timeline + stats) and
# Nsight Compute (kernel-level counters), then summarize the transfer vs
# compute breakdown into Markdown.
#
# Requirements: a CUDA GPU, nsys and ncu on PATH, and the repo built
# (cmake -B build && cmake --build build -j).
#
# Environment overrides:
#   OUTDIR     output directory            (default: results/)
#   BENCH_BIN  benchmark binary            (default: ./build/bench_gemm)
#   M N K BATCH ITERS IMPL  benchmark parameters
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTDIR=${OUTDIR:-"${REPO_ROOT}/results"}
BENCH_BIN=${BENCH_BIN:-"${REPO_ROOT}/build/bench_gemm"}
M=${M:-1024}
N=${N:-1024}
K=${K:-1024}
BATCH=${BATCH:-8}
ITERS=${ITERS:-10}
IMPL=${IMPL:-tiled}

mkdir -p "${OUTDIR}"

if [[ ! -x "${BENCH_BIN}" ]]; then
  echo "error: ${BENCH_BIN} not found; run: cmake -B build && cmake --build build -j" >&2
  exit 1
fi
if ! command -v nsys >/dev/null 2>&1; then
  echo "error: nsys not on PATH (install Nsight Systems / CUDA toolkit)" >&2
  exit 1
fi

echo "nsys: profiling ${BENCH_BIN} (M=${M} N=${N} K=${K} batch=${BATCH} iters=${ITERS} impl=${IMPL})"
nsys profile --stats=true --force-overwrite=true \
  -o "${OUTDIR}/nsys_report" \
  "${BENCH_BIN}" --m "${M}" --n "${N}" --k "${K}" --batch "${BATCH}" \
    --iters "${ITERS}" --impl "${IMPL}"

if command -v ncu >/dev/null 2>&1; then
  echo "ncu: profiling the tiled kernel"
  ncu --set full --force-overwrite \
    --export "${OUTDIR}/ncu_tiled" \
    "${BENCH_BIN}" --m "${M}" --n "${N}" --k "${K}" --batch "${BATCH}" \
      --iters 1 --impl tiled
else
  echo "warning: ncu not found; skipping kernel-level profiling" >&2
fi

echo "parsing nsys report"
python3 "${SCRIPT_DIR}/parse_nsys.py" "${OUTDIR}/nsys_report.sqlite" \
  -o "${OUTDIR}/nsys_breakdown.md"

echo "profiling artifacts in ${OUTDIR}:"
for f in "${OUTDIR}"/nsys_report* "${OUTDIR}"/ncu_tiled* "${OUTDIR}"/nsys_breakdown.md; do
  [[ -e "$f" ]] && printf '%s\n' "$f"
done
