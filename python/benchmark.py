#!/usr/bin/env python3
"""Fair batched-GEMM benchmark: NumPy CPU vs tiled CUDA vs cuBLAS.

Methodology
-----------
* GPU kernel-only and end-to-end (H2D + kernel + D2H) times are measured with
  cudaEvents inside bench_gemm: 10 untimed warmup runs, then the median of
  --runs timed runs (default 60).
* The CPU baseline is np.matmul on float32; the BLAS implementation, version
  and thread count actually used are printed.
* GFLOP/s is 2*B*M*N*K / time. "% peak" uses the printed FP32 peak for this
  GPU. cuBLAS is the primary GPU baseline; the tiled kernel is also reported
  as a percentage of cuBLAS throughput.
* Writes results/benchmark.csv and results/speedup.png.
"""

import argparse
import csv
import io
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import threadpoolctl
except ImportError:  # pragma: no cover - only used for reporting
    threadpoolctl = None

# README shapes plus shapes where a 32x32x32 shared-memory tiled kernel has
# little or no reuse: non-multiples of 32 and skinny matrices.
DEFAULT_SHAPES = [
    (2, 256, 256, 256),
    (4, 512, 512, 512),
    (8, 1024, 1024, 1024),
    (1, 2048, 2048, 2048),
    (3, 257, 129, 65),
    (8, 257, 257, 257),
    (4, 1000, 33, 31),
    (2, 1000, 1, 1000),
    (1, 1, 1000, 1000),
    (1, 1000, 1000, 1),
]

# Tesla T4: 40 SMs * 64 FP32 lanes/SM * 2 FLOP/lane/cycle * 1.590 GHz.
DEFAULT_PEAK_TFLOPS = 40 * 64 * 2 * 1.590e9 / 1e12


def parse_shape(text):
    parts = text.split("x")
    if len(parts) != 4:
        raise ValueError("shape must be BxMxNxK")
    return tuple(int(p) for p in parts)


def median(values):
    return statistics.median(values)


def gpu_name():
    try:
        return (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
            )
            .strip()
            .splitlines()[0]
        )
    except Exception:
        return "unknown GPU"


def blas_description():
    if threadpoolctl is None:
        return "threadpoolctl unavailable; see np.show_config()", "unknown"
    pools = [p for p in threadpoolctl.threadpool_info() if p.get("user_api") == "blas"]
    if not pools:
        return "no BLAS pool reported", "unknown"
    p = pools[0]
    desc = (
        f"{p.get('internal_api', '?')} {p.get('version', '?')} "
        f"({Path(p.get('filepath', '?')).name}, {p.get('threading_layer', '?')})"
    )
    return desc, p.get("num_threads", "unknown")


def run_bench_gemm(bench, shape, impl, runs, warmup):
    batch, m, n, k = shape
    cmd = [
        str(bench),
        "--m",
        str(m),
        "--n",
        str(n),
        "--k",
        str(k),
        "--batch",
        str(batch),
        "--iters",
        str(runs),
        "--warmup",
        str(warmup),
        "--impl",
        impl,
        "--skip-check",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows = list(csv.reader(io.StringIO(proc.stdout)))
    rows = [r for r in rows if len(r) >= 10 and r[0] != "impl"]
    if not rows:
        raise RuntimeError(f"no benchmark row from: {' '.join(cmd)}\n{proc.stdout}")
    header = next(
        csv.reader(
            io.StringIO(
                [line for line in proc.stdout.splitlines() if line.startswith("impl,")][
                    0
                ]
            )
        )
    )
    values = dict(zip(header, rows[0]))
    out = {}
    for key, value in values.items():
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def time_numpy(a, b, runs, warmup):
    for _ in range(warmup):
        np.matmul(a, b)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        np.matmul(a, b)
        times.append((time.perf_counter() - start) * 1e3)
    return median(times)


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key, value in row.items():
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
            rows.append(row)
    return rows


def write_plot(rows, peak_tflops, runs, png_path):
    labels = [r["shape"] for r in rows]
    x = np.arange(len(labels))
    width = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    tiled_g = [r["tiled_kernel_gflops"] for r in rows]
    cublas_g = [r["cublas_kernel_gflops"] for r in rows]
    ax1.bar(x - width / 2, tiled_g, width, label="tiled", color="tab:blue")
    ax1.bar(x + width / 2, cublas_g, width, label="cuBLAS", color="tab:orange")
    ax1.axhline(
        peak_tflops * 1000.0,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"FP32 peak {peak_tflops:.2f} TFLOP/s",
    )
    ax1.set_ylabel("kernel-only GFLOP/s")
    ax1.set_title("Batched GEMM kernel throughput (higher is better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.25)

    pct = [r["tiled_pct_cublas_kernel"] for r in rows]
    colors = ["tab:red" if v < 50 else "tab:green" for v in pct]
    bars = ax2.bar(x, pct, color=colors)
    ax2.axhline(
        100.0, color="black", linestyle="--", linewidth=1, label="cuBLAS = 100%"
    )
    ax2.bar_label(bars, fmt="%.0f%%", fontsize=8)
    ax2.set_ylim(0, max(110.0, max(pct) * 1.15))
    ax2.set_ylabel("tiled GFLOP/s as % of cuBLAS")
    ax2.set_title("Tiled kernel vs cuBLAS (kernel-only, same FLOPs)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"{gpu_name()}: tiled vs cuBLAS, median of {runs} runs "
        f"(FP32 peak {peak_tflops:.2f} TFLOP/s)"
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)


def print_markdown(rows):
    print("\nMarkdown table (kernel-only GPU times; e2e includes H2D+D2H):")
    print(
        "| Shape (B x M x N x K) | CPU ms (NumPy) | tiled kernel ms | tiled e2e ms | "
        "tiled GFLOP/s | tiled % peak | cuBLAS kernel ms | cuBLAS e2e ms | "
        "cuBLAS GFLOP/s | cuBLAS % peak | tiled % of cuBLAS |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['shape']} | {r['cpu_ms']:.3f} | {r['tiled_kernel_ms']:.3f} | "
            f"{r['tiled_e2e_ms']:.3f} | {r['tiled_kernel_gflops']:.1f} | "
            f"{r['tiled_pct_peak']:.1f}% | {r['cublas_kernel_ms']:.3f} | "
            f"{r['cublas_e2e_ms']:.3f} | {r['cublas_kernel_gflops']:.1f} | "
            f"{r['cublas_pct_peak']:.1f}% | {r['tiled_pct_cublas_kernel']:.1f}% |"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        metavar="BxMxNxK",
        help="add a shape (may be repeated)",
    )
    parser.add_argument("--runs", type=int, default=60, help="timed runs (default 60)")
    parser.add_argument(
        "--warmup", type=int, default=10, help="warmup runs (default 10)"
    )
    parser.add_argument(
        "--no-defaults",
        action="store_true",
        help="benchmark only the shapes passed with --shape",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate speedup.png from an existing benchmark.csv",
    )
    parser.add_argument("--outdir", default="results", help="output directory")
    parser.add_argument(
        "--bench",
        default=str(Path(__file__).resolve().parents[1] / "build" / "bench_gemm"),
        help="path to bench_gemm",
    )
    parser.add_argument(
        "--peak-tflops",
        type=float,
        default=DEFAULT_PEAK_TFLOPS,
        help="FP32 peak in TFLOP/s (default: Tesla T4 8.1408)",
    )
    args = parser.parse_args()

    if args.runs < 50:
        print("warning: fewer than 50 timed runs requested", file=sys.stderr)
    if args.warmup < 10:
        print("warning: fewer than 10 warmup runs requested", file=sys.stderr)

    if args.plot_only:
        outdir = Path(args.outdir)
        csv_path = outdir / "benchmark.csv"
        png_path = outdir / "speedup.png"
        if not csv_path.exists():
            print(f"error: {csv_path} not found", file=sys.stderr)
            return 1
        rows = load_rows(csv_path)
        write_plot(rows, args.peak_tflops, args.runs, png_path)
        print(f"wrote {png_path}")
        print_markdown(rows)
        return 0

    bench = Path(args.bench)
    if not bench.exists():
        print(f"error: bench_gemm not found at {bench}", file=sys.stderr)
        return 1

    shapes = ([] if args.no_defaults else list(DEFAULT_SHAPES)) + [
        parse_shape(s) for s in args.shape
    ]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "benchmark.csv"
    png_path = outdir / "speedup.png"

    threads = os.cpu_count() or 1
    _blas_limiter = None
    if threadpoolctl is not None:
        # Keep the controller alive so the limit applies for the whole run.
        _blas_limiter = threadpoolctl.threadpool_limits(limits=threads, user_api="blas")
    blas_name, blas_threads = blas_description()

    print(f"GPU: {gpu_name()}")
    print(
        f"FP32 peak used: {args.peak_tflops:.4f} TFLOP/s "
        f"({args.peak_tflops * 1000.0:.1f} GFLOP/s)"
    )
    print(f"NumPy: {np.__version__} on {platform.platform()}")
    print(f"CPU BLAS: {blas_name}")
    print(
        f"CPU threads used: {blas_threads} "
        f"(requested {threads}, os.cpu_count={os.cpu_count()})"
    )
    print(f"Timing: {args.warmup} warmup + median of {args.runs} timed runs")
    print()

    rng = np.random.default_rng(42)
    rows = []
    for batch, m, n, k in shapes:
        shape = (batch, m, n, k)
        label = f"{batch}x{m}x{n}x{k}"
        print(f"[{label}] measuring...", flush=True)
        a = rng.uniform(-1.0, 1.0, (batch, m, k)).astype(np.float32)
        b = rng.uniform(-1.0, 1.0, (batch, k, n)).astype(np.float32)
        flops = 2.0 * batch * m * n * k
        cpu_ms = time_numpy(a, b, args.runs, args.warmup)
        tiled = run_bench_gemm(bench, shape, "tiled", args.runs, args.warmup)
        cublas = run_bench_gemm(bench, shape, "cublas", args.runs, args.warmup)

        peak_gflops = args.peak_tflops * 1000.0
        row = {
            "shape": label,
            "batch": batch,
            "m": m,
            "n": n,
            "k": k,
            "flops": flops,
            "cpu_ms": cpu_ms,
            "cpu_gflops": flops / cpu_ms / 1e6,
            "tiled_h2d_ms": tiled["h2d_ms"],
            "tiled_kernel_ms": tiled["kernel_ms"],
            "tiled_e2e_ms": tiled["total_ms"],
            "tiled_kernel_gflops": tiled["kernel_gflops"],
            "tiled_e2e_gflops": tiled["e2e_gflops"],
            "tiled_pct_peak": 100.0 * tiled["kernel_gflops"] / peak_gflops,
            "tiled_speedup_cpu_kernel": cpu_ms / tiled["kernel_ms"],
            "tiled_speedup_cpu_e2e": cpu_ms / tiled["total_ms"],
            "cublas_h2d_ms": cublas["h2d_ms"],
            "cublas_kernel_ms": cublas["kernel_ms"],
            "cublas_e2e_ms": cublas["total_ms"],
            "cublas_kernel_gflops": cublas["kernel_gflops"],
            "cublas_e2e_gflops": cublas["e2e_gflops"],
            "cublas_pct_peak": 100.0 * cublas["kernel_gflops"] / peak_gflops,
            "cublas_speedup_cpu_kernel": cpu_ms / cublas["kernel_ms"],
            "cublas_speedup_cpu_e2e": cpu_ms / cublas["total_ms"],
        }
        row["tiled_pct_cublas_kernel"] = (
            100.0 * row["tiled_kernel_gflops"] / row["cublas_kernel_gflops"]
        )
        row["tiled_pct_cublas_e2e"] = (
            100.0 * row["tiled_e2e_gflops"] / row["cublas_e2e_gflops"]
        )
        rows.append(row)
        print(
            f"  cpu {cpu_ms:9.3f} ms | tiled kernel {row['tiled_kernel_ms']:9.3f} ms "
            f"({row['tiled_kernel_gflops']:7.1f} GFLOP/s, "
            f"{row['tiled_pct_peak']:5.1f}% peak) | "
            f"cuBLAS kernel {row['cublas_kernel_ms']:9.3f} ms "
            f"({row['cublas_kernel_gflops']:7.1f} GFLOP/s, "
            f"{row['cublas_pct_peak']:5.1f}% peak) | "
            f"tiled/cuBLAS {row['tiled_pct_cublas_kernel']:5.1f}%",
            flush=True,
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {csv_path}")

    write_plot(rows, args.peak_tflops, args.runs, png_path)
    print(f"wrote {png_path}")
    print_markdown(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
