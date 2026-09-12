#!/usr/bin/env python3
"""Summarize an Nsight Systems report (nsys_report.sqlite) as Markdown.

Emits a table of GPU kernel time vs memcpy HtoD vs memcpy DtoH vs total wall
time, with percentages. Kernel time is the sum of kernel durations; wall time
spans the first to the last recorded activity (so gaps / CPU time show up in
the "Other" row).

Usage:
    python3 scripts/parse_nsys.py <nsys_report.sqlite> [-o out.md]
"""
import argparse
import sqlite3
import sys

KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"
MEMCPY_TABLE = "CUPTI_ACTIVITY_KIND_MEMCPY"

# CUPTI memcpy kinds: 1 = HtoD, 2 = DtoH
KIND_HTOD = 1
KIND_DTOH = 2


def read_table(db, table):
    try:
        return db.execute(f"SELECT start, end FROM {table}").fetchall()
    except sqlite3.OperationalError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="path to nsys_report.sqlite")
    parser.add_argument(
        "-o", "--output", help="write Markdown to this file (default: stdout)"
    )
    args = parser.parse_args()

    try:
        db = sqlite3.connect(args.report)
    except sqlite3.Error as ex:
        print(f"error: cannot open {args.report}: {ex}", file=sys.stderr)
        return 1

    kernel_ns = 0.0
    htod_ns = 0.0
    dtoh_ns = 0.0
    starts = []
    ends = []

    kernel_rows = read_table(db, KERNEL_TABLE)
    if kernel_rows is None:
        print(
            f"error: table {KERNEL_TABLE} not found in {args.report}; "
            "is this an nsys sqlite report?",
            file=sys.stderr,
        )
        return 1
    for start, end in kernel_rows:
        kernel_ns += end - start
        starts.append(start)
        ends.append(end)

    try:
        memcpy_rows = db.execute(
            f"SELECT start, end, copyKind FROM {MEMCPY_TABLE}"
        ).fetchall()
    except sqlite3.OperationalError:
        print(
            f"warning: table {MEMCPY_TABLE} not found; transfer columns will be 0",
            file=sys.stderr,
        )
        memcpy_rows = []
    for start, end, kind in memcpy_rows:
        starts.append(start)
        ends.append(end)
        if kind == KIND_HTOD:
            htod_ns += end - start
        elif kind == KIND_DTOH:
            dtoh_ns += end - start

    if not starts:
        print("error: report contains no kernel activity", file=sys.stderr)
        return 1

    total_ns = max(ends) - min(starts)
    copies_ns = htod_ns + dtoh_ns
    other_ns = max(0.0, total_ns - kernel_ns - copies_ns)

    def ms(ns: float) -> str:
        return f"{ns / 1e6:.3f}"

    def pct(ns: float) -> str:
        return f"{100.0 * ns / total_ns:.2f}" if total_ns > 0 else "0.00"

    rows = [
        ("GPU kernels", kernel_ns),
        ("Memcpy HtoD", htod_ns),
        ("Memcpy DtoH", dtoh_ns),
        ("Other (CPU / gaps)", other_ns),
        ("**Total**", total_ns),
    ]

    lines = [
        f"# Nsight Systems breakdown ({args.report})",
        "",
        "| Category | Time (ms) | % of wall time |",
        "|---|---|---|",
    ]
    for label, ns in rows:
        lines.append(f"| {label} | {ms(ns)} | {pct(ns)} |")
    lines.append("")
    lines.append("_Kernel time is the summed duration of all GPU kernels; wall time "
                 "spans the first to the last recorded activity._")
    text = "\n".join(lines) + "\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
