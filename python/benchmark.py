#!/usr/bin/env python3
"""Benchmark batched GEMM: NumPy (CPU) vs gpuinfer (GPU).

Runs a set of shapes through np.matmul and gpuinfer.batched_gemm, asserts the
GPU result matches NumPy (rtol=1e-3), prints a table of CPU ms / GPU ms /
speedup, and writes results/benchmark.csv plus a matplotlib bar chart to
results/speedup.png.
"""

import argparse
import csv
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gpuinfer

DEFAULT_SHAPES = [
    (2, 256, 256, 256),
    (4, 512, 512, 512),
    (8, 1024, 1024, 1024),
    (1, 2048, 2048, 2048),
]


def parse_shape(text):
    parts = text.split("x")
    if len(parts) != 4:
        raise ValueError("shape must be BxMxNxK")
    return tuple(int(p) for p in parts)


def time_cpu(a, b, iters):
    np.matmul(a, b)  # warmup
    best = float("inf")
    for _ in range(iters):
        start = time.perf_counter()
        np.matmul(a, b)
        best = min(best, (time.perf_counter() - start) * 1e3)
    return best


def time_gpu(a, b, iters):
    gpuinfer.batched_gemm(a, b)  # warmup
    best = float("inf")
    for _ in range(iters):
        start = time.perf_counter()
        gpuinfer.batched_gemm(a, b)
        best = min(best, (time.perf_counter() - start) * 1e3)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        metavar="BxMxNxK",
        help="add a shape (may be repeated)",
    )
    parser.add_argument("--iters", type=int, default=5, help="timed iterations")
    parser.add_argument(
        "--outdir", default="results", help="output directory (default results)"
    )
    args = parser.parse_args()

    if not gpuinfer.cuda_available():
        print("error: no CUDA GPU available", file=sys.stderr)
        return 1

    shapes = list(DEFAULT_SHAPES)
    for text in args.shape:
        shapes.append(parse_shape(text))

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "benchmark.csv")
    png_path = os.path.join(args.outdir, "speedup.png")

    rng = np.random.default_rng(42)
    rows = []
    for batch, m, n, k in shapes:
        a = rng.uniform(-1.0, 1.0, (batch, m, k)).astype(np.float32)
        b = rng.uniform(-1.0, 1.0, (batch, k, n)).astype(np.float32)
        cpu_result = np.matmul(a, b)
        gpu_result = gpuinfer.batched_gemm(a, b)
        if not np.allclose(gpu_result, cpu_result, rtol=1e-3, atol=1e-4):
            raise AssertionError(
                f"shape {batch}x{m}x{n}x{k}: GPU result does not match NumPy "
                f"(max abs diff {np.max(np.abs(gpu_result - cpu_result)):.3e})"
            )
        cpu_ms = time_cpu(a, b, max(1, args.iters // 2))
        gpu_ms = time_gpu(a, b, args.iters)
        speedup = cpu_ms / gpu_ms if gpu_ms > 0 else float("inf")
        rows.append(
            {
                "shape": f"{batch}x{m}x{n}x{k}",
                "batch": batch,
                "m": m,
                "n": n,
                "k": k,
                "cpu_ms": cpu_ms,
                "gpu_ms": gpu_ms,
                "speedup": speedup,
            }
        )

    print()
    print(f"{'shape':>16}  {'cpu_ms':>10}  {'gpu_ms':>10}  {'speedup':>8}")
    for row in rows:
        print(
            f"{row['shape']:>16}  {row['cpu_ms']:10.3f}  "
            f"{row['gpu_ms']:10.3f}  {row['speedup']:8.2f}x"
        )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["shape", "batch", "m", "n", "k", "cpu_ms", "gpu_ms", "speedup"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    labels = [row["shape"] for row in rows]
    speedups = [row["speedup"] for row in rows]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, speedups, color="tab:blue")
    plt.bar_label(bars, fmt="%.2fx")
    plt.ylabel("speedup vs NumPy")
    plt.title("gpuinfer batched GEMM speedup vs NumPy")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    print(f"wrote {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
