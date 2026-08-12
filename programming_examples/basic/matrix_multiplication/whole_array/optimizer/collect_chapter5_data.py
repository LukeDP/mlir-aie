#!/usr/bin/env python3
"""
Chapter 5 hardware-data collector for the final Super-Tuner framework.

Place this script in the same directory as:
    - main_launcher_wBenchmark.py
    - tiling_optimizer.py

The script:
  * builds the fixed Option-A workload campaign;
  * de-duplicates workloads while preserving all category labels;
  * records geometric and analytically valid design-space sizes;
  * runs the fixed AMD baseline when the 32x32x32 / 4x8 mapping is geometrically legal;
  * runs the final Super-Tuner exactly as implemented in main_launcher_wBenchmark.py;
  * re-benchmarks the selected optimized tile with FINAL_ITERATIONS;
  * stores descriptor-derived transfer volumes (NOT measured DDR traffic);
  * writes raw stdout/stderr logs for every final benchmark;
  * writes one row per final benchmark iteration;
  * supports resuming an interrupted campaign.

No plotting is performed here. Plot scripts should only read the generated CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import main_launcher_wBenchmark as launcher
from tiling_optimizer import (
    DEFAULT_COLS,
    DEFAULT_ROWS,
    filter_by_memory,
    find_candidates,
)


# =============================================================================
# Frozen experimental constants
# =============================================================================

BASELINE_TILE = (32, 32, 32)
BASELINE_ROWS = 4
BASELINE_COLS = 8

OPTIMIZED_ROWS = DEFAULT_ROWS   # final framework: 2
OPTIMIZED_COLS = DEFAULT_COLS   # final framework: 8

FINAL_ITERATIONS = launcher.FINAL_ITERATIONS
TUNING_ITERATIONS = launcher.TUNING_ITERATIONS
DE_POPSIZE = launcher.DE_POPSIZE
DE_MAXITER = launcher.DE_MAXITER
DE_SEED = launcher.DE_SEED
EXHAUSTIVE_THRESHOLD = launcher.EXHAUSTIVE_THRESHOLD


# =============================================================================
# Option-A Chapter 5 workload campaign
# =============================================================================

N_SCALING = [
    (128, 1024, 128,  "i16"),
    (128, 1024, 256,  "i16"),
    (128, 1024, 512,  "i16"),
    (128, 1024, 1024, "i16"),
    (128, 1024, 2048, "i16"),
    (128, 1024, 4096, "i16"),
]

K_SCALING = [
    (128, 64,   1024, "i16"),
    (128, 128,  1024, "i16"),
    (128, 256,  1024, "i16"),
    (128, 512,  1024, "i16"),
    (128, 1024, 1024, "i16"),
    (128, 2048, 1024, "i16"),
    (128, 4096, 1024, "i16"),
]

M_SCALING = [
    (64,   128, 1024, "i16"),
    (128,  128, 1024, "i16"),
    (256,  128, 1024, "i16"),
    (512,  128, 1024, "i16"),
    (1024, 128, 1024, "i16"),
    (2048, 128, 1024, "i16"),
    (4096, 128, 1024, "i16"),
]

SQUARE = [
    (128,  128,  128,  "i16"),
    (256,  256,  256,  "i16"),
    (512,  512,  512,  "i16"),
    (1024, 1024, 1024, "i16"),
    (2048, 2048, 2048, "i16"),
]

REPRESENTATIVE = [
    (512, 512, 512,  "i16"),
    (512, 768, 768,  "bf16"),
    (512, 512, 2048, "i16"),
    (512, 2048, 512, "i16"),
]

CATEGORY_WORKLOADS = {
    "N_scaling": N_SCALING,
    "K_scaling": K_SCALING,
    "M_scaling": M_SCALING,
    "square": SQUARE,
    "representative": REPRESENTATIVE,
}


# =============================================================================
# CSV schemas
# =============================================================================

MASTER_FIELDS = [
    "workload_key",
    "categories",
    "timestamp_utc",
    "M",
    "K",
    "N",
    "dtype",
    "total_ops",

    "m_candidate_count",
    "k_candidate_count",
    "n_candidate_count",
    "geometric_candidate_count",
    "valid_tile_count",
    "pruning_pct",

    "de_popsize",
    "de_maxiter",
    "de_seed",
    "exhaustive_threshold",
    "tuning_iterations",
    "final_iterations",

    "search_mode",
    "unique_hil_evaluations",
    "evaluated_fraction",
    "tuning_wall_time_s",

    "baseline_valid_geometry",
    "baseline_m",
    "baseline_k",
    "baseline_n",
    "baseline_rows",
    "baseline_cols",
    "baseline_gops",
    "baseline_std_gops",
    "baseline_min_gops",
    "baseline_max_gops",
    "baseline_benchmark_wall_time_s",

    "opt_m",
    "opt_k",
    "opt_n",
    "opt_rows",
    "opt_cols",
    "opt_l1_bytes",
    "opt_l2_bytes",
    "opt_search_gops",
    "opt_final_gops",
    "opt_final_std_gops",
    "opt_final_min_gops",
    "opt_final_max_gops",
    "opt_final_benchmark_wall_time_s",

    "throughput_speedup",

    "baseline_A_desc_bytes",
    "baseline_B_desc_bytes",
    "baseline_C_desc_bytes",
    "baseline_total_desc_bytes",
    "opt_A_desc_bytes",
    "opt_B_desc_bytes",
    "opt_C_desc_bytes",
    "opt_total_desc_bytes",

    "A_desc_reduction_factor",
    "B_desc_reduction_factor",
    "C_desc_reduction_factor",
    "total_desc_reduction_factor",

    "baseline_R_A",
    "opt_R_A",

    "status",
    "error_message",
]

MEASUREMENT_FIELDS = [
    "workload_key",
    "categories",
    "M",
    "K",
    "N",
    "dtype",
    "mode",
    "m",
    "k",
    "n",
    "rows",
    "cols",
    "iteration",
    "gops",
]


# =============================================================================
# Utility helpers
# =============================================================================

def workload_key(M: int, K: int, N: int, dtype: str) -> str:
    return f"{M}x{K}x{N}_{dtype}"


def utc_now_string() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den is None or den == 0:
        return None
    return num / den


def safe_round(value: Optional[float], digits: int = 6):
    if value is None:
        return ""
    return round(float(value), digits)


def build_campaign() -> List[Dict]:
    """
    De-duplicate the logical campaign while preserving every category membership.
    """
    category_map: Dict[Tuple[int, int, int, str], List[str]] = defaultdict(list)

    for category, workloads in CATEGORY_WORKLOADS.items():
        for item in workloads:
            if category not in category_map[item]:
                category_map[item].append(category)

    rows = []
    for (M, K, N, dtype), categories in category_map.items():
        rows.append(
            {
                "M": M,
                "K": K,
                "N": N,
                "dtype": dtype,
                "categories": ";".join(categories),
                "workload_key": workload_key(M, K, N, dtype),
            }
        )

    # Deterministic ordering: INT16 campaign first in logical size order, BF16 after.
    rows.sort(
        key=lambda r: (
            0 if r["dtype"] == "i16" else 1,
            r["M"],
            r["K"],
            r["N"],
            r["workload_key"],
        )
    )
    return rows


def write_manifest(path: Path, campaign: Sequence[Dict]) -> None:
    fields = ["workload_key", "categories", "M", "K", "N", "dtype"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(campaign)


def read_completed_keys(master_csv: Path) -> set:
    if not master_csv.exists():
        return set()

    completed = set()
    with master_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "complete":
                completed.add(row["workload_key"])
    return completed


def append_csv_row(path: Path, fields: Sequence[str], row: Dict) -> None:
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


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


# =============================================================================
# Design-space accounting
# =============================================================================

def analyze_design_space(M: int, K: int, N: int, dtype: str) -> Dict:
    m_list, k_list, n_list, _ = find_candidates(
        M, K, N, dtype=dtype, cols=OPTIMIZED_COLS
    )

    geometric_count = len(m_list) * len(k_list) * len(n_list)

    valid = filter_by_memory(
        m_list,
        k_list,
        n_list,
        M,
        K,
        N,
        dtype=dtype,
        cols=OPTIMIZED_COLS,
    )

    valid_count = len(valid)
    pruning_pct = (
        100.0 * (geometric_count - valid_count) / geometric_count
        if geometric_count > 0
        else 0.0
    )

    by_tile = {
        (int(m), int(k), int(n)): {
            "l1_bytes": int(l1),
            "l2_bytes": int(l2),
        }
        for m, k, n, l1, l2 in valid
    }

    return {
        "m_candidate_count": len(m_list),
        "k_candidate_count": len(k_list),
        "n_candidate_count": len(n_list),
        "geometric_candidate_count": geometric_count,
        "valid_tile_count": valid_count,
        "pruning_pct": pruning_pct,
        "valid_by_tile": by_tile,
    }


# =============================================================================
# Baseline validity and descriptor-derived transfer model
# =============================================================================

def baseline_geometry_is_valid(M: int, K: int, N: int, dtype: str) -> bool:
    """
    Baseline whole_array mapping:
      * fixed core tile 32x32x32
      * 4 active rows
      * 8 active columns

    We reject geometrically impossible baseline cases instead of recording a fake 0 GOPS.
    """
    m, k, n = BASELINE_TILE

    if M % (m * BASELINE_ROWS) != 0:
        return False
    if K % k != 0:
        return False
    if N % (n * BASELINE_COLS) != 0:
        return False

    # Both evaluated datatypes satisfy the baseline 32x32x32 vector alignment.
    return dtype in {"i16", "bf16"}


def descriptor_transfer_bytes(
    M: int,
    K: int,
    N: int,
    dtype: str,
    m: int,
    k: int,
    n: int,
    rows: int,
    cols: int,
) -> Dict[str, int]:
    """
    Descriptor-derived transfer-volume model used in the final thesis.

        T_A = M*K*E_in * N/(n*C)
        T_B = K*N*E_in * M/(m*R)
        T_C = M*N*E_C

    These quantities are analytically derived from the generated mapping/descriptor
    structure. They are NOT measurements of physical DDR traffic.
    """
    if M % (m * rows) != 0:
        raise ValueError("M is not divisible by m*rows for this mapping.")
    if N % (n * cols) != 0:
        raise ValueError("N is not divisible by n*cols for this mapping.")
    if K % k != 0:
        raise ValueError("K is not divisible by k for this mapping.")

    input_bytes = 2
    c_bytes = 2 if dtype == "bf16" else 4

    I_N = N // (n * cols)
    I_M = M // (m * rows)

    A = M * K * input_bytes * I_N
    B = K * N * input_bytes * I_M
    C = M * N * c_bytes

    return {
        "A": int(A),
        "B": int(B),
        "C": int(C),
        "total": int(A + B + C),
        "R_A": int(I_N),
    }


# =============================================================================
# Hardware benchmark execution
# =============================================================================

ITER_RE = re.compile(
    r"Iteration\s+(\d+)/(\d+)\.\.\.\s+([0-9]+(?:\.[0-9]+)?)\s+GOPS"
)
AVG_RE = re.compile(
    r"Final Average Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s+GOPS"
)


def run_final_benchmark(
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
    mode: str,
    iterations: int = FINAL_ITERATIONS,
) -> Dict:
    """
    Compile once and run the Makefile benchmark for `iterations` hardware executions.

    Returns the final average plus the individual per-iteration GOPS values parsed
    from stderr. Raw stdout/stderr are saved in logs_dir.
    """
    if mode not in {"baseline", "optimized"}:
        raise ValueError("mode must be 'baseline' or 'optimized'")

    use_poc = "0" if mode == "baseline" else "1"
    opt_perf = "0" if mode == "baseline" else "1"
    dtype_out = "bf16" if dtype == "bf16" else "i32"

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
        f"use_poc={use_poc}",
        f"ITERATIONS={iterations}",
        f"opt_perf={opt_perf}",
        f"dtype_in={dtype}",
        f"dtype_out={dtype_out}",
    ]

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0

    tag = f"{workload_key(M, K, N, dtype)}_{mode}_{m}x{k}x{n}"
    stdout_path = logs_dir / f"{tag}.stdout.txt"
    stderr_path = logs_dir / f"{tag}.stderr.txt"
    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"{mode} benchmark failed for {tag}. "
            f"See {stdout_path.name} / {stderr_path.name}."
        )

    avg_match = AVG_RE.search(result.stdout)
    if not avg_match:
        raise RuntimeError(
            f"Could not parse Final Average Throughput for {tag}."
        )
    average_gops = float(avg_match.group(1))

    per_iteration = [
        (int(m.group(1)), float(m.group(3)))
        for m in ITER_RE.finditer(result.stderr)
    ]

    values = np.array([v for _, v in per_iteration], dtype=float)

    # The final average remains the Makefile's own reported value.
    stats = {
        "average_gops": average_gops,
        "wall_time_s": elapsed,
        "iterations": per_iteration,
        "std_gops": float(values.std(ddof=0)) if len(values) else None,
        "min_gops": float(values.min()) if len(values) else None,
        "max_gops": float(values.max()) if len(values) else None,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    return stats


# =============================================================================
# One-workload collection
# =============================================================================

def collect_one(
    campaign_row: Dict,
    project_root: Path,
    master_csv: Path,
    measurements_csv: Path,
    logs_dir: Path,
) -> None:
    M = int(campaign_row["M"])
    K = int(campaign_row["K"])
    N = int(campaign_row["N"])
    dtype = campaign_row["dtype"]
    categories = campaign_row["categories"]
    key = campaign_row["workload_key"]

    print("\n" + "=" * 78)
    print(f"[Chapter 5] {key} | categories={categories}")
    print("=" * 78)

    row = {field: "" for field in MASTER_FIELDS}
    row.update(
        {
            "workload_key": key,
            "categories": categories,
            "timestamp_utc": utc_now_string(),
            "M": M,
            "K": K,
            "N": N,
            "dtype": dtype,
            "total_ops": 2 * M * K * N,
            "de_popsize": DE_POPSIZE,
            "de_maxiter": DE_MAXITER,
            "de_seed": DE_SEED,
            "exhaustive_threshold": EXHAUSTIVE_THRESHOLD,
            "tuning_iterations": TUNING_ITERATIONS,
            "final_iterations": FINAL_ITERATIONS,
            "baseline_m": BASELINE_TILE[0],
            "baseline_k": BASELINE_TILE[1],
            "baseline_n": BASELINE_TILE[2],
            "baseline_rows": BASELINE_ROWS,
            "baseline_cols": BASELINE_COLS,
            "opt_rows": OPTIMIZED_ROWS,
            "opt_cols": OPTIMIZED_COLS,
            "status": "incomplete",
            "error_message": "",
        }
    )

    try:
        # ---------------------------------------------------------------------
        # 1. Analytical design-space accounting
        # ---------------------------------------------------------------------
        ds = analyze_design_space(M, K, N, dtype)
        row.update(
            {
                "m_candidate_count": ds["m_candidate_count"],
                "k_candidate_count": ds["k_candidate_count"],
                "n_candidate_count": ds["n_candidate_count"],
                "geometric_candidate_count": ds["geometric_candidate_count"],
                "valid_tile_count": ds["valid_tile_count"],
                "pruning_pct": safe_round(ds["pruning_pct"], 4),
            }
        )

        if ds["valid_tile_count"] == 0:
            raise RuntimeError("No analytically valid optimized tiles.")

        # ---------------------------------------------------------------------
        # 2. Baseline final benchmark (only if baseline geometry is legal)
        # ---------------------------------------------------------------------
        baseline_valid = baseline_geometry_is_valid(M, K, N, dtype)
        row["baseline_valid_geometry"] = int(baseline_valid)

        baseline_desc = None
        baseline_stats = None

        if baseline_valid:
            print("[Chapter 5] Running baseline final benchmark...")
            baseline_stats = run_final_benchmark(
                project_root=project_root,
                logs_dir=logs_dir,
                M=M,
                K=K,
                N=N,
                dtype=dtype,
                m=BASELINE_TILE[0],
                k=BASELINE_TILE[1],
                n=BASELINE_TILE[2],
                mode="baseline",
            )

            row.update(
                {
                    "baseline_gops": safe_round(baseline_stats["average_gops"]),
                    "baseline_std_gops": safe_round(baseline_stats["std_gops"]),
                    "baseline_min_gops": safe_round(baseline_stats["min_gops"]),
                    "baseline_max_gops": safe_round(baseline_stats["max_gops"]),
                    "baseline_benchmark_wall_time_s": safe_round(
                        baseline_stats["wall_time_s"], 3
                    ),
                }
            )

            for iteration, gops in baseline_stats["iterations"]:
                append_csv_row(
                    measurements_csv,
                    MEASUREMENT_FIELDS,
                    {
                        "workload_key": key,
                        "categories": categories,
                        "M": M,
                        "K": K,
                        "N": N,
                        "dtype": dtype,
                        "mode": "baseline_final",
                        "m": BASELINE_TILE[0],
                        "k": BASELINE_TILE[1],
                        "n": BASELINE_TILE[2],
                        "rows": BASELINE_ROWS,
                        "cols": BASELINE_COLS,
                        "iteration": iteration,
                        "gops": gops,
                    },
                )

            baseline_desc = descriptor_transfer_bytes(
                M,
                K,
                N,
                dtype,
                BASELINE_TILE[0],
                BASELINE_TILE[1],
                BASELINE_TILE[2],
                BASELINE_ROWS,
                BASELINE_COLS,
            )
            row.update(
                {
                    "baseline_A_desc_bytes": baseline_desc["A"],
                    "baseline_B_desc_bytes": baseline_desc["B"],
                    "baseline_C_desc_bytes": baseline_desc["C"],
                    "baseline_total_desc_bytes": baseline_desc["total"],
                    "baseline_R_A": baseline_desc["R_A"],
                }
            )
        else:
            print(
                "[Chapter 5] Baseline 32x32x32 / 4x8 mapping is geometrically "
                "invalid for this workload; recording N/A rather than 0 GOPS."
            )

        # ---------------------------------------------------------------------
        # 3. Final Super-Tuner HIL search
        # ---------------------------------------------------------------------
        print("[Chapter 5] Running final Super-Tuner...")
        t0 = time.perf_counter()
        best = launcher.run_super_tuner(
            M,
            K,
            N,
            dtype,
            seed=DE_SEED,
        )
        tuning_elapsed = time.perf_counter() - t0

        opt_tile = (
            int(best["m"]),
            int(best["k"]),
            int(best["n"]),
        )

        if opt_tile not in ds["valid_by_tile"]:
            raise RuntimeError(
                f"Super-Tuner returned {opt_tile}, which is not in D_valid."
            )

        mem = ds["valid_by_tile"][opt_tile]

        row.update(
            {
                "search_mode": best["search_mode"],
                "unique_hil_evaluations": int(best["evaluated_tiles"]),
                "evaluated_fraction": safe_round(
                    best["evaluated_tiles"] / best["valid_tiles"], 6
                ),
                "tuning_wall_time_s": safe_round(tuning_elapsed, 3),
                "opt_m": opt_tile[0],
                "opt_k": opt_tile[1],
                "opt_n": opt_tile[2],
                "opt_l1_bytes": mem["l1_bytes"],
                "opt_l2_bytes": mem["l2_bytes"],
                "opt_search_gops": safe_round(best["gops"]),
            }
        )

        # ---------------------------------------------------------------------
        # 4. Dedicated optimized final benchmark (20 iterations)
        # ---------------------------------------------------------------------
        print("[Chapter 5] Running optimized final benchmark...")
        opt_stats = run_final_benchmark(
            project_root=project_root,
            logs_dir=logs_dir,
            M=M,
            K=K,
            N=N,
            dtype=dtype,
            m=opt_tile[0],
            k=opt_tile[1],
            n=opt_tile[2],
            mode="optimized",
        )

        row.update(
            {
                "opt_final_gops": safe_round(opt_stats["average_gops"]),
                "opt_final_std_gops": safe_round(opt_stats["std_gops"]),
                "opt_final_min_gops": safe_round(opt_stats["min_gops"]),
                "opt_final_max_gops": safe_round(opt_stats["max_gops"]),
                "opt_final_benchmark_wall_time_s": safe_round(
                    opt_stats["wall_time_s"], 3
                ),
            }
        )

        for iteration, gops in opt_stats["iterations"]:
            append_csv_row(
                measurements_csv,
                MEASUREMENT_FIELDS,
                {
                    "workload_key": key,
                    "categories": categories,
                    "M": M,
                    "K": K,
                    "N": N,
                    "dtype": dtype,
                    "mode": "optimized_final",
                    "m": opt_tile[0],
                    "k": opt_tile[1],
                    "n": opt_tile[2],
                    "rows": OPTIMIZED_ROWS,
                    "cols": OPTIMIZED_COLS,
                    "iteration": iteration,
                    "gops": gops,
                },
            )

        # ---------------------------------------------------------------------
        # 5. Descriptor-derived transfer-volume model
        # ---------------------------------------------------------------------
        opt_desc = descriptor_transfer_bytes(
            M,
            K,
            N,
            dtype,
            opt_tile[0],
            opt_tile[1],
            opt_tile[2],
            OPTIMIZED_ROWS,
            OPTIMIZED_COLS,
        )

        row.update(
            {
                "opt_A_desc_bytes": opt_desc["A"],
                "opt_B_desc_bytes": opt_desc["B"],
                "opt_C_desc_bytes": opt_desc["C"],
                "opt_total_desc_bytes": opt_desc["total"],
                "opt_R_A": opt_desc["R_A"],
            }
        )

        if baseline_stats is not None and baseline_desc is not None:
            row.update(
                {
                    "throughput_speedup": safe_round(
                        safe_ratio(
                            opt_stats["average_gops"],
                            baseline_stats["average_gops"],
                        ),
                        6,
                    ),
                    "A_desc_reduction_factor": safe_round(
                        safe_ratio(baseline_desc["A"], opt_desc["A"]), 6
                    ),
                    "B_desc_reduction_factor": safe_round(
                        safe_ratio(baseline_desc["B"], opt_desc["B"]), 6
                    ),
                    "C_desc_reduction_factor": safe_round(
                        safe_ratio(baseline_desc["C"], opt_desc["C"]), 6
                    ),
                    "total_desc_reduction_factor": safe_round(
                        safe_ratio(
                            baseline_desc["total"],
                            opt_desc["total"],
                        ),
                        6,
                    ),
                }
            )

        row["status"] = "complete"

    except Exception as exc:
        row["status"] = "failed"
        row["error_message"] = str(exc)
        print(f"[Chapter 5][ERROR] {key}: {exc}", file=sys.stderr)

    # Always append a row so failures are auditable and the campaign state is visible.
    append_csv_row(master_csv, MASTER_FIELDS, row)

    print(
        f"[Chapter 5] Stored {key}: status={row['status']} "
        f"-> {master_csv}"
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect the final Chapter 5 hardware dataset."
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help=(
            "Path to the whole_array directory containing the Makefile. "
            "If omitted, auto-detect current directory or parent."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="chapter5_data",
        help="Output directory (default: chapter5_data).",
    )
    parser.add_argument(
        "--category",
        choices=list(CATEGORY_WORKLOADS.keys()),
        default=None,
        help="Run only workloads belonging to one category.",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default=None,
        help=(
            "Run one workload key only, e.g. 512x512x2048_i16. "
            "Useful for testing before the full campaign."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run at most N selected workloads (debug/testing helper).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip workloads already marked complete in the master CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the manifest and print selected workloads without using hardware.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = resolve_project_root(args.project_root)

    output_dir = Path(args.output_dir).expanduser().resolve()
    raw_dir = output_dir / "raw"
    logs_dir = output_dir / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = raw_dir / "chapter5_workload_manifest.csv"
    master_csv = raw_dir / "chapter5_workload_results.csv"
    measurements_csv = raw_dir / "chapter5_final_measurements.csv"

    campaign = build_campaign()
    write_manifest(manifest_csv, campaign)

    selected = campaign

    if args.category:
        selected = [
            row
            for row in selected
            if args.category in row["categories"].split(";")
        ]

    if args.workload:
        selected = [
            row for row in selected if row["workload_key"] == args.workload
        ]
        if not selected:
            raise SystemExit(f"Unknown workload key: {args.workload}")

    if args.limit is not None:
        selected = selected[: args.limit]

    if not args.no_resume:
        completed = read_completed_keys(master_csv)
        selected = [
            row for row in selected if row["workload_key"] not in completed
        ]

    print(f"[Chapter 5] Project root: {project_root}")
    print(f"[Chapter 5] Output dir:   {output_dir}")
    print(f"[Chapter 5] Campaign has {len(campaign)} unique workloads.")
    print(f"[Chapter 5] Selected this run: {len(selected)}")

    for row in selected:
        print(
            f"  - {row['workload_key']:24s} "
            f"[{row['categories']}]"
        )

    if args.dry_run:
        print("[Chapter 5] Dry run complete; no hardware execution performed.")
        return

    for row in selected:
        collect_one(
            row,
            project_root,
            master_csv,
            measurements_csv,
            logs_dir,
        )

    print("\n" + "=" * 78)
    print("[Chapter 5] Campaign invocation completed.")
    print(f"Manifest:     {manifest_csv}")
    print(f"Master data:  {master_csv}")
    print(f"Measurements: {measurements_csv}")
    print(f"Logs:         {logs_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
