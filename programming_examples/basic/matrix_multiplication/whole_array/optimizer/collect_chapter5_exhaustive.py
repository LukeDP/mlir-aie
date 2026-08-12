#!/usr/bin/env python3
"""
Final exhaustive Hardware-in-the-Loop validation collector for Chapter 5.

This script is intentionally separate from the main Chapter 5 workload campaign.
Its purpose is to build the empirical exhaustive reference used in Section 5.2.

It evaluates every analytically valid tile for exactly three validation workloads:

    1) 512 x 512 x 512   INT16
    2) 512 x 768 x 768   BF16
    3) 512 x 512 x 2048  INT16

The expected final valid-space sizes with the frozen framework are:

    21 + 28 + 7 = 56 tiles.

For every valid tile the script:
  * records candidate-space accounting;
  * records modeled L1/L2 footprints;
  * executes the optimized mapping on the physical NPU;
  * uses TUNING_ITERATIONS from the frozen final launcher (15);
  * stores the averaged Hardware_GOPS;
  * stores every individual hardware iteration in a second CSV;
  * stores raw stdout/stderr logs;
  * supports resume after interruption.

The resulting exhaustive CSV is the empirical reference for the later
Projected-DE offline replay. It is NOT a mathematical proof of global
optimality outside the analytically admitted design space.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import main_launcher_wBenchmark as launcher
from tiling_optimizer import (
    DEFAULT_COLS,
    DEFAULT_ROWS,
    filter_by_memory,
    find_candidates,
)


# =============================================================================
# Frozen validation campaign
# =============================================================================

VALIDATION_WORKLOADS = [
    {
        "M": 512,
        "K": 512,
        "N": 512,
        "dtype": "i16",
        "label": "BALANCED_INT16",
    },
    {
        "M": 512,
        "K": 768,
        "N": 768,
        "dtype": "bf16",
        "label": "TRANSFORMER_BF16",
    },
    {
        "M": 512,
        "K": 512,
        "N": 2048,
        "dtype": "i16",
        "label": "OUTPUT_STRESS_INT16",
    },
]

ROWS = DEFAULT_ROWS
COLS = DEFAULT_COLS
TUNING_ITERATIONS = launcher.TUNING_ITERATIONS

EXPECTED_VALID_COUNTS = {
    (512, 512, 512, "i16"): 21,
    (512, 768, 768, "bf16"): 28,
    (512, 512, 2048, "i16"): 7,
}


# =============================================================================
# CSV schemas
# =============================================================================

TILE_FIELDS = [
    "timestamp_utc",
    "workload_label",
    "M",
    "K",
    "N",
    "Data_Type",

    "m_candidate_count",
    "k_candidate_count",
    "n_candidate_count",
    "Geometric_Candidate_Count",
    "Valid_Tile_Count",
    "Pruning_Percentage",

    "m",
    "k",
    "n",
    "L1_Bytes",
    "L2_Bytes",

    "Tuning_Iterations",
    "Hardware_GOPS",
    "Hardware_STD_GOPS",
    "Hardware_Min_GOPS",
    "Hardware_Max_GOPS",
    "Benchmark_Wall_Time_s",

    "status",
    "error_message",
]

MEASUREMENT_FIELDS = [
    "workload_label",
    "M",
    "K",
    "N",
    "Data_Type",
    "m",
    "k",
    "n",
    "iteration",
    "GOPS",
]

SUMMARY_FIELDS = [
    "workload_label",
    "M",
    "K",
    "N",
    "Data_Type",
    "m_candidate_count",
    "k_candidate_count",
    "n_candidate_count",
    "Geometric_Candidate_Count",
    "Valid_Tile_Count",
    "Pruning_Percentage",
    "Expected_Valid_Tile_Count",
    "Expected_Count_Match",
]


# =============================================================================
# Helpers
# =============================================================================

ITER_RE = re.compile(
    r"Iteration\s+(\d+)/(\d+)\.\.\.\s+([0-9]+(?:\.[0-9]+)?)\s+GOPS"
)
AVG_RE = re.compile(
    r"Final Average Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s+GOPS"
)


def utc_now_string() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def workload_key(M: int, K: int, N: int, dtype: str) -> str:
    return f"{M}x{K}x{N}_{dtype}"


def tile_key(M: int, K: int, N: int, dtype: str, m: int, k: int, n: int) -> str:
    return f"{workload_key(M, K, N, dtype)}_{m}x{k}x{n}"


def safe_round(value: Optional[float], digits: int = 6):
    if value is None:
        return ""
    return round(float(value), digits)


def resolve_project_root(explicit: Optional[str]) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
    else:
        here = Path(__file__).resolve().parent
        if (here / "Makefile").exists():
            root = here
        elif (here.parent / "Makefile").exists():
            root = here.parent
        else:
            raise FileNotFoundError(
                "Could not locate the whole_array Makefile. "
                "Pass --project-root /path/to/whole_array."
            )

    if not (root / "Makefile").exists():
        raise FileNotFoundError(f"No Makefile found in project root: {root}")

    return root


def append_csv_row(path: Path, fields: Sequence[str], row: Dict) -> None:
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_completed_tiles(path: Path) -> set:
    if not path.exists():
        return set()

    completed = set()

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "complete":
                continue

            completed.add(
                (
                    int(row["M"]),
                    int(row["K"]),
                    int(row["N"]),
                    row["Data_Type"],
                    int(row["m"]),
                    int(row["k"]),
                    int(row["n"]),
                )
            )

    return completed


# =============================================================================
# Analytical design-space enumeration
# =============================================================================

def enumerate_valid_tiles(M: int, K: int, N: int, dtype: str) -> Dict:
    """
    Candidate counts are measured AFTER divisor generation, spatial divisibility,
    and datatype-specific microkernel alignment.

    Therefore Geometric_Candidate_Count is the geometrically/microkernel-admissible
    design space before the subsequent analytical feasibility filtering.
    """
    m_candidates, k_candidates, n_candidates, _ = find_candidates(
        M,
        K,
        N,
        dtype=dtype,
        cols=COLS,
    )

    geometric_count = (
        len(m_candidates)
        * len(k_candidates)
        * len(n_candidates)
    )

    valid_tiles = filter_by_memory(
        m_candidates,
        k_candidates,
        n_candidates,
        M,
        K,
        N,
        dtype=dtype,
        cols=COLS,
    )

    valid_count = len(valid_tiles)

    pruning_percentage = (
        100.0 * (geometric_count - valid_count) / geometric_count
        if geometric_count > 0
        else 0.0
    )

    return {
        "m_candidates": list(m_candidates),
        "k_candidates": list(k_candidates),
        "n_candidates": list(n_candidates),
        "m_candidate_count": len(m_candidates),
        "k_candidate_count": len(k_candidates),
        "n_candidate_count": len(n_candidates),
        "geometric_count": geometric_count,
        "valid_tiles": valid_tiles,
        "valid_count": valid_count,
        "pruning_percentage": pruning_percentage,
    }


# =============================================================================
# Hardware execution
# =============================================================================

def run_hardware_tile(
    *,
    project_root: Path,
    logs_dir: Path,
    M: int,
    K: int,
    N: int,
    dtype: str,
    m: int,
    k: int,
    n: int,
) -> Dict:
    """
    Execute one analytically valid optimized tile using the same HIL fitness
    conditions used by the final Super-Tuner: opt_perf=1, use_poc=1,
    TUNING_ITERATIONS hardware iterations.
    """
    dtype_out = "bf16" if dtype == "bf16" else "i32"

    # Keep each tile compilation independent.
    launcher.clean_build_artifacts()

    cmd = [
        "make",
        "run_plot",
        f"M={M}",
        f"K={K}",
        f"N={N}",
        f"m={m}",
        f"k={k}",
        f"n={n}",
        "use_poc=1",
        f"ITERATIONS={TUNING_ITERATIONS}",
        "opt_perf=1",
        f"dtype_in={dtype}",
        f"dtype_out={dtype_out}",
    ]

    tag = tile_key(M, K, N, dtype, m, k, n)

    print(f"    [HIL] {m}x{k}x{n} ... ", end="", flush=True)

    t0 = time.perf_counter()

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    elapsed = time.perf_counter() - t0

    stdout_path = logs_dir / f"{tag}.stdout.txt"
    stderr_path = logs_dir / f"{tag}.stderr.txt"

    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)

    if result.returncode != 0:
        print("FAILED")
        raise RuntimeError(
            f"HIL execution failed for {tag}. "
            f"See {stdout_path.name} and {stderr_path.name}."
        )

    avg_match = AVG_RE.search(result.stdout)

    if not avg_match:
        print("PARSE FAILED")
        raise RuntimeError(
            f"Could not parse Final Average Throughput for {tag}."
        )

    average_gops = float(avg_match.group(1))

    per_iteration = [
        (int(match.group(1)), float(match.group(3)))
        for match in ITER_RE.finditer(result.stderr)
    ]

    values = np.array(
        [gops for _, gops in per_iteration],
        dtype=float,
    )

    print(f"{average_gops:.3f} GOPS")

    return {
        "average_gops": average_gops,
        "std_gops": float(values.std(ddof=0)) if len(values) else None,
        "min_gops": float(values.min()) if len(values) else None,
        "max_gops": float(values.max()) if len(values) else None,
        "iterations": per_iteration,
        "wall_time_s": elapsed,
        "stdout_log": stdout_path,
        "stderr_log": stderr_path,
    }


# =============================================================================
# Summary generation
# =============================================================================

def write_filter_summary(
    summary_csv: Path,
    analytical_results: List[Dict],
) -> None:
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()

        for result in analytical_results:
            expected = EXPECTED_VALID_COUNTS.get(
                (
                    result["M"],
                    result["K"],
                    result["N"],
                    result["dtype"],
                )
            )

            writer.writerow(
                {
                    "workload_label": result["label"],
                    "M": result["M"],
                    "K": result["K"],
                    "N": result["N"],
                    "Data_Type": result["dtype"],
                    "m_candidate_count": result["m_candidate_count"],
                    "k_candidate_count": result["k_candidate_count"],
                    "n_candidate_count": result["n_candidate_count"],
                    "Geometric_Candidate_Count": result["geometric_count"],
                    "Valid_Tile_Count": result["valid_count"],
                    "Pruning_Percentage": safe_round(
                        result["pruning_percentage"], 6
                    ),
                    "Expected_Valid_Tile_Count": expected,
                    "Expected_Count_Match": int(
                        expected == result["valid_count"]
                    ),
                }
            )


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect the final exhaustive Chapter 5 HIL validation dataset."
        )
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help=(
            "Path to the whole_array directory containing the Makefile. "
            "If omitted, current directory / parent are auto-detected."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="chapter5_data",
        help="Output directory (default: chapter5_data).",
    )

    parser.add_argument(
        "--workload",
        type=str,
        default=None,
        help=(
            "Run only one validation workload, e.g. "
            "512x512x512_i16."
        ),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Do not skip tiles already marked complete in the exhaustive CSV."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Enumerate and print the final valid spaces without executing hardware."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = resolve_project_root(args.project_root)

    output_root = Path(args.output_dir).expanduser().resolve()
    validation_dir = output_root / "validation"
    logs_dir = validation_dir / "logs_exhaustive"

    validation_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    tiles_csv = validation_dir / "chapter5_exhaustive_tiles.csv"
    measurements_csv = (
        validation_dir
        / "chapter5_exhaustive_measurements.csv"
    )
    summary_csv = (
        validation_dir
        / "chapter5_filter_summary.csv"
    )

    selected_workloads = VALIDATION_WORKLOADS

    if args.workload is not None:
        selected_workloads = [
            item
            for item in VALIDATION_WORKLOADS
            if workload_key(
                item["M"],
                item["K"],
                item["N"],
                item["dtype"],
            )
            == args.workload
        ]

        if not selected_workloads:
            raise SystemExit(
                f"Unknown validation workload: {args.workload}"
            )

    print(f"[Exhaustive] Project root: {project_root}")
    print(f"[Exhaustive] Output root:  {output_root}")
    print(f"[Exhaustive] Active array: {ROWS} rows x {COLS} columns")
    print(f"[Exhaustive] HIL iterations per tile: {TUNING_ITERATIONS}")

    analytical_results = []

    # -------------------------------------------------------------------------
    # First enumerate every residual design space and validate expected sizes.
    # -------------------------------------------------------------------------
    for workload in selected_workloads:
        M = workload["M"]
        K = workload["K"]
        N = workload["N"]
        dtype = workload["dtype"]

        ds = enumerate_valid_tiles(M, K, N, dtype)

        record = {
            **workload,
            **ds,
        }

        analytical_results.append(record)

        expected = EXPECTED_VALID_COUNTS.get((M, K, N, dtype))

        print("\n" + "=" * 80)
        print(
            f"[Exhaustive] {M}x{K}x{N} ({dtype}) "
            f"[{workload['label']}]"
        )
        print(
            f"  m candidates: {ds['m_candidate_count']} "
            f"{ds['m_candidates']}"
        )
        print(
            f"  k candidates: {ds['k_candidate_count']} "
            f"{ds['k_candidates']}"
        )
        print(
            f"  n candidates: {ds['n_candidate_count']} "
            f"{ds['n_candidates']}"
        )
        print(
            f"  geometric/microkernel combinations: "
            f"{ds['geometric_count']}"
        )
        print(
            f"  analytically valid tiles: {ds['valid_count']}"
        )
        print(
            f"  pruning: {ds['pruning_percentage']:.2f}%"
        )

        if expected is not None and ds["valid_count"] != expected:
            raise RuntimeError(
                f"Frozen-framework validation failed for "
                f"{M}x{K}x{N}_{dtype}: expected {expected} valid tiles, "
                f"found {ds['valid_count']}."
            )

    write_filter_summary(summary_csv, analytical_results)

    total_valid = sum(
        result["valid_count"]
        for result in analytical_results
    )

    print("\n" + "=" * 80)
    print(
        f"[Exhaustive] Total selected valid tiles: {total_valid}"
    )

    if args.dry_run:
        print(
            "[Exhaustive] Dry run complete. "
            "No hardware execution performed."
        )
        print(f"[Exhaustive] Filter summary: {summary_csv}")
        return

    completed_tiles = (
        set()
        if args.no_resume
        else read_completed_tiles(tiles_csv)
    )

    pending = 0

    for result in analytical_results:
        M = result["M"]
        K = result["K"]
        N = result["N"]
        dtype = result["dtype"]

        for m, k, n, l1_bytes, l2_bytes in result["valid_tiles"]:
            key = (
                M,
                K,
                N,
                dtype,
                int(m),
                int(k),
                int(n),
            )

            if key not in completed_tiles:
                pending += 1

    print(
        f"[Exhaustive] Already complete: "
        f"{total_valid - pending}/{total_valid}"
    )
    print(f"[Exhaustive] Pending HIL evaluations: {pending}")

    # -------------------------------------------------------------------------
    # Exhaustive physical evaluation
    # -------------------------------------------------------------------------
    for result in analytical_results:
        M = result["M"]
        K = result["K"]
        N = result["N"]
        dtype = result["dtype"]
        label = result["label"]

        print("\n" + "=" * 80)
        print(
            f"[Exhaustive] HIL sweep: {M}x{K}x{N} ({dtype}) "
            f"| {result['valid_count']} valid tiles"
        )
        print("=" * 80)

        for m, k, n, l1_bytes, l2_bytes in result["valid_tiles"]:
            m = int(m)
            k = int(k)
            n = int(n)

            completed_key = (
                M,
                K,
                N,
                dtype,
                m,
                k,
                n,
            )

            if completed_key in completed_tiles:
                print(
                    f"    [Resume] {m}x{k}x{n} already complete."
                )
                continue

            row = {
                field: ""
                for field in TILE_FIELDS
            }

            row.update(
                {
                    "timestamp_utc": utc_now_string(),
                    "workload_label": label,
                    "M": M,
                    "K": K,
                    "N": N,
                    "Data_Type": dtype,

                    "m_candidate_count": result[
                        "m_candidate_count"
                    ],
                    "k_candidate_count": result[
                        "k_candidate_count"
                    ],
                    "n_candidate_count": result[
                        "n_candidate_count"
                    ],
                    "Geometric_Candidate_Count": result[
                        "geometric_count"
                    ],
                    "Valid_Tile_Count": result[
                        "valid_count"
                    ],
                    "Pruning_Percentage": safe_round(
                        result["pruning_percentage"], 6
                    ),

                    "m": m,
                    "k": k,
                    "n": n,
                    "L1_Bytes": int(l1_bytes),
                    "L2_Bytes": int(l2_bytes),

                    "Tuning_Iterations": TUNING_ITERATIONS,
                    "status": "incomplete",
                    "error_message": "",
                }
            )

            try:
                stats = run_hardware_tile(
                    project_root=project_root,
                    logs_dir=logs_dir,
                    M=M,
                    K=K,
                    N=N,
                    dtype=dtype,
                    m=m,
                    k=k,
                    n=n,
                )

                row.update(
                    {
                        "Hardware_GOPS": safe_round(
                            stats["average_gops"], 6
                        ),
                        "Hardware_STD_GOPS": safe_round(
                            stats["std_gops"], 6
                        ),
                        "Hardware_Min_GOPS": safe_round(
                            stats["min_gops"], 6
                        ),
                        "Hardware_Max_GOPS": safe_round(
                            stats["max_gops"], 6
                        ),
                        "Benchmark_Wall_Time_s": safe_round(
                            stats["wall_time_s"], 3
                        ),
                        "status": "complete",
                    }
                )

                for iteration, gops in stats["iterations"]:
                    append_csv_row(
                        measurements_csv,
                        MEASUREMENT_FIELDS,
                        {
                            "workload_label": label,
                            "M": M,
                            "K": K,
                            "N": N,
                            "Data_Type": dtype,
                            "m": m,
                            "k": k,
                            "n": n,
                            "iteration": iteration,
                            "GOPS": gops,
                        },
                    )

            except Exception as exc:
                row["status"] = "failed"
                row["error_message"] = str(exc)

                print(
                    f"    [ERROR] {m}x{k}x{n}: {exc}",
                    file=sys.stderr,
                )

            append_csv_row(
                tiles_csv,
                TILE_FIELDS,
                row,
            )

    print("\n" + "=" * 80)
    print("[Exhaustive] Campaign invocation completed.")
    print(f"[Exhaustive] Tiles:        {tiles_csv}")
    print(f"[Exhaustive] Measurements: {measurements_csv}")
    print(f"[Exhaustive] Summary:      {summary_csv}")
    print(f"[Exhaustive] Logs:         {logs_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
