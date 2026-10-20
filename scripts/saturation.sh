#!/usr/bin/env bash
# Measure scheduler throughput under increasing offered load and at saturation.
#
# The 20 jobs/s run in the README is arrival-rate-limited: 19.36 jobs/s
# completed against 20 jobs/s offered, so it says nothing about the scheduler's
# ceiling. This script runs the same synthetic workload at several Poisson
# rates and then in open-loop saturation mode (--saturation submits every job
# immediately, so the backlog never drains).
#
# Usage:
#   scripts/saturation.sh
#   JOBS=2000 RATES="50 200 saturated" scripts/saturation.sh
#
# Environment overrides:
#   SCHED_BIN   scheduler_daemon binary (default: ./build/scheduler_daemon)
#   JOBS        jobs per run (default: 1000)
#   WORKERS     worker threads (default: 2)
#   MAX_BATCH   jobs per packed batch (default: 8)
#   IMPL        tiled | cublas (default: tiled)
#   MARGIN_MB   GPU memory safety margin (default: 256)
#   RATES       space-separated Poisson rates plus the literal "saturated"
#               (default: "20 100 500 saturated")
#   OUTDIR      artifact directory (default: results/saturation)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCHED_BIN=${SCHED_BIN:-"${REPO_ROOT}/build/scheduler_daemon"}
JOBS=${JOBS:-1000}
WORKERS=${WORKERS:-2}
MAX_BATCH=${MAX_BATCH:-8}
IMPL=${IMPL:-tiled}
MARGIN_MB=${MARGIN_MB:-256}
OUTDIR=${OUTDIR:-"${REPO_ROOT}/results/saturation"}

if [[ ! -x "${SCHED_BIN}" ]]; then
  echo "error: ${SCHED_BIN} not found; run: cmake -B build && cmake --build build -j" >&2
  exit 1
fi

mkdir -p "${OUTDIR}"
SUMMARY="${OUTDIR}/summary.csv"
printf '%s\n' \
  'mode,offered_rate_jobs_per_s,jobs,workers,max_batch,impl,submitted,completed,failed,wall_s,throughput_jobs_per_s,p50_ms,p95_ms,p99_ms,mean_gpu_utilization_pct' \
  > "${SUMMARY}"

extract() {
  local pattern="$1"
  local file="$2"
  sed -nE "${pattern}" "${file}" | tail -n 1
}

run_one() {
  local mode="$1"
  local offered="$2"
  local label
  if [[ "${mode}" == "saturated" ]]; then
    label="saturated"
  else
    label="rate_${offered}"
  fi

  local log="${OUTDIR}/${label}.log"
  local metrics="${OUTDIR}/${label}_jobs.csv"
  local args=(
    --jobs "${JOBS}"
    --workers "${WORKERS}"
    --max-batch "${MAX_BATCH}"
    --impl "${IMPL}"
    --margin-mb "${MARGIN_MB}"
    --http-port 0
    --metrics-csv "${metrics}"
  )
  if [[ "${mode}" == "saturated" ]]; then
    args+=(--saturation)
  else
    args+=(--arrival-rate "${offered}")
  fi

  echo "==> ${label} (jobs=${JOBS} workers=${WORKERS} max_batch=${MAX_BATCH})" >&2
  "${SCHED_BIN}" "${args[@]}" >"${log}" 2>&1

  local submitted completed failed wall throughput p50 p95 p99 util
  submitted=$(extract 's/^\[scheduler\] submitted=([0-9]+).*/\1/p' "${log}")
  completed=$(extract 's/^\[scheduler\] submitted=[0-9]+ completed=([0-9]+).*/\1/p' "${log}")
  failed=$(extract 's/^\[scheduler\] submitted=[0-9]+ completed=[0-9]+ failed=([0-9]+).*/\1/p' "${log}")
  wall=$(extract 's/.*wall=([0-9.]+)s.*/\1/p' "${log}")
  throughput=$(extract 's/^\[scheduler\] throughput=([0-9.]+).*/\1/p' "${log}")
  p50=$(extract 's/.*p50=([0-9.]+).*/\1/p' "${log}")
  p95=$(extract 's/.*p95=([0-9.]+).*/\1/p' "${log}")
  p99=$(extract 's/.*p99=([0-9.]+).*/\1/p' "${log}")
  util=$(extract 's/^\[scheduler\] mean gpu utilization: ([0-9.]+)%.*/\1/p' "${log}")

  if [[ -z "${throughput}" || -z "${completed}" ]]; then
    echo "error: could not parse scheduler output in ${log}" >&2
    cat "${log}" >&2
    return 1
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${mode}" "${offered}" "${JOBS}" "${WORKERS}" "${MAX_BATCH}" "${IMPL}" \
    "${submitted}" "${completed}" "${failed}" "${wall}" "${throughput}" \
    "${p50}" "${p95}" "${p99}" "${util}" >>"${SUMMARY}"

  printf '    %s jobs/s, p50/p95/p99 %s/%s/%s ms, GPU %s%%\n' \
    "${throughput}" "${p50}" "${p95}" "${p99}" "${util}" >&2
}

read -r -a rates <<<"${RATES:-20 100 500 saturated}"
for rate in "${rates[@]}"; do
  if [[ "${rate}" == "saturated" ]]; then
    run_one saturated 0
  else
    run_one poisson "${rate}"
  fi
done

echo
echo "wrote ${SUMMARY}"
column -t -s, "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
