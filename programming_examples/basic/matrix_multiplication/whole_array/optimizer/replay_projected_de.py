#!/usr/bin/env python3
"""
Offline Projected-DE replay for Chapter 5 validation.

Input:
    chapter5_data/validation/chapter5_exhaustive_tiles.csv

Outputs:
    chapter5_data/validation/chapter5_de_runs.csv
    chapter5_data/validation/chapter5_de_summary.csv
    chapter5_data/validation/chapter5_de_parameter_sweep.csv

The replay uses the hardware-measured GOPS values from the exhaustive HIL
campaign as a lookup table. No hardware is invoked here.

Important methodological interpretation:
  * The "reference" is the best configuration OBSERVED in the exhaustive HIL
    evaluation over the analytically admitted finite design space.
  * It is not called a mathematical or ground-truth optimum.
  * Only workloads with more than EXHAUSTIVE_THRESHOLD valid tiles are replayed,
    because smaller spaces use exhaustive HIL in the real final framework.
  * The Projected-DE path matches the final framework:
        - normalized log2 coordinates
        - Euclidean nearest-valid-tile projection
        - differential_evolution(strategy="best1bin")
        - tol=0, atol=0
        - polish=False
        - deterministic seed
  * The final deployed budget is popsize=4, maxiter=2.
  * A compact parameter sweep is evaluated for:
        (3,2), (4,2), (4,3), (6,2)
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


# =============================================================================
# Frozen framework / validation settings
# =============================================================================

EXHAUSTIVE_THRESHOLD = 12
N_SEEDS = 100

FINAL_POPSIZE = 4
FINAL_MAXITER = 2

PARAMETER_SWEEP = [
    (3, 2),
    (4, 2),
    (4, 3),
    (6, 2),
]

NEAR_OPTIMAL_1PCT = 1.0
NEAR_OPTIMAL_2PCT = 2.0


# =============================================================================
# Output schemas
# =============================================================================

RUN_FIELDS = [
    "workload_label",
    "M",
    "K",
    "N",
    "Data_Type",
    "valid_tiles",

    "popsize",
    "maxiter",
    "seed",

    "reference_m",
    "reference_k",
    "reference_n",
    "reference_gops",

    "selected_m",
    "selected_k",
    "selected_n",
    "selected_gops",

    "exact_match",
    "gap_pct",
    "near_optimal_1pct",
    "near_optimal_2pct",

    "unique_tiles_evaluated",
    "evaluated_fraction",
]

SUMMARY_FIELDS = [
    "workload_label",
    "M",
    "K",
    "N",
    "Data_Type",
    "valid_tiles",

    "popsize",
    "maxiter",
    "seeds",

    "reference_m",
    "reference_k",
    "reference_n",
    "reference_gops",

    "exact_match_count",
    "exact_match_rate_pct",

    "near_optimal_1pct_count",
    "near_optimal_1pct_rate_pct",

    "near_optimal_2pct_count",
    "near_optimal_2pct_rate_pct",

    "mean_gap_pct",
    "median_gap_pct",
    "std_gap_pct",
    "worst_gap_pct",

    "avg_unique_tiles_evaluated",
    "min_unique_tiles_evaluated",
    "max_unique_tiles_evaluated",
    "avg_evaluated_fraction",
    "avg_evaluated_fraction_pct",
]

SWEEP_FIELDS = SUMMARY_FIELDS


# =============================================================================
# Helpers
# =============================================================================

WORKLOAD_COLUMNS = [
    "workload_label",
    "M",
    "K",
    "N",
    "Data_Type",
]


def safe_round(value, digits: int = 6):
    return round(float(value), digits)


def workload_key(row) -> str:
    return (
        f"{int(row['M'])}x{int(row['K'])}x{int(row['N'])}_"
        f"{row['Data_Type']}"
    )


def validate_input(df: pd.DataFrame) -> None:
    required = {
        "workload_label",
        "M",
        "K",
        "N",
        "Data_Type",
        "m",
        "k",
        "n",
        "Hardware_GOPS",
        "status",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Input exhaustive CSV is missing columns: "
            + ", ".join(missing)
        )

    if len(df) == 0:
        raise ValueError("Input exhaustive CSV is empty.")

    incomplete = df[df["status"] != "complete"]
    if not incomplete.empty:
        raise ValueError(
            f"Input contains {len(incomplete)} non-complete exhaustive rows."
        )

    if (df["Hardware_GOPS"] <= 0).any():
        bad = df[df["Hardware_GOPS"] <= 0]
        raise ValueError(
            f"Input contains {len(bad)} non-positive Hardware_GOPS values."
        )

    duplicate_mask = df.duplicated(
        subset=[
            "M",
            "K",
            "N",
            "Data_Type",
            "m",
            "k",
            "n",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise ValueError(
            "Duplicate workload/tile rows found in exhaustive CSV."
        )


# =============================================================================
# Projected-DE replay
# =============================================================================

def prepare_workload(workload_df: pd.DataFrame) -> Dict:
    """
    Build the exact discrete lookup/projection representation used by the replay.
    """
    workload_df = workload_df.copy().reset_index(drop=True)

    tile_array = workload_df[["m", "k", "n"]].to_numpy(dtype=int)
    gops_array = workload_df["Hardware_GOPS"].to_numpy(dtype=float)

    log_tiles = np.log2(tile_array.astype(float))

    coord_min = log_tiles.min(axis=0)
    coord_max = log_tiles.max(axis=0)

    coord_range = np.where(
        coord_max > coord_min,
        coord_max - coord_min,
        1.0,
    )

    normalized_tiles = (
        log_tiles - coord_min
    ) / coord_range

    hardware_results = {
        tuple(int(x) for x in tile): float(gops)
        for tile, gops in zip(tile_array, gops_array)
    }

    reference_tile = max(
        hardware_results,
        key=hardware_results.get,
    )
    reference_gops = hardware_results[reference_tile]

    return {
        "tile_array": tile_array,
        "gops_array": gops_array,
        "normalized_tiles": normalized_tiles,
        "hardware_results": hardware_results,
        "reference_tile": reference_tile,
        "reference_gops": reference_gops,
        "valid_tiles": len(tile_array),
    }


def replay_one_seed(
    prepared: Dict,
    *,
    seed: int,
    popsize: int,
    maxiter: int,
) -> Dict:
    """
    Replay one Projected-DE search using measured exhaustive HIL GOPS as fitness.

    The returned selected tile is the best physically measured tile ENCOUNTERED
    by the replay, not merely decode(result.x). This matches the final framework's
    "best measured tile encountered" semantics.
    """
    tile_array = prepared["tile_array"]
    normalized_tiles = prepared["normalized_tiles"]
    hardware_results = prepared["hardware_results"]

    reference_tile = prepared["reference_tile"]
    reference_gops = prepared["reference_gops"]

    visited_tiles = set()
    best_tile = None
    best_gops = -math.inf

    def decode_candidate(x):
        x = np.asarray(x, dtype=float)

        distances = np.sum(
            (normalized_tiles - x) ** 2,
            axis=1,
        )

        nearest_index = int(np.argmin(distances))

        return tuple(
            int(v)
            for v in tile_array[nearest_index]
        )

    def objective_function(x):
        nonlocal best_tile, best_gops

        tile = decode_candidate(x)
        gops = hardware_results[tile]

        visited_tiles.add(tile)

        if gops > best_gops:
            best_tile = tile
            best_gops = gops

        return -gops

    differential_evolution(
        objective_function,
        bounds=[
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
        ],
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=0.0,
        atol=0.0,
        seed=seed,
        polish=False,
    )

    if best_tile is None:
        raise RuntimeError(
            "Projected-DE replay completed without evaluating a tile."
        )

    gap_pct = (
        (reference_gops - best_gops)
        / reference_gops
        * 100.0
    )

    # Numerical noise should not create a tiny negative gap.
    if abs(gap_pct) < 1e-12:
        gap_pct = 0.0

    return {
        "selected_tile": best_tile,
        "selected_gops": best_gops,
        "reference_tile": reference_tile,
        "reference_gops": reference_gops,
        "exact_match": best_tile == reference_tile,
        "gap_pct": gap_pct,
        "near_optimal_1pct": gap_pct <= NEAR_OPTIMAL_1PCT,
        "near_optimal_2pct": gap_pct <= NEAR_OPTIMAL_2PCT,
        "unique_tiles_evaluated": len(visited_tiles),
        "evaluated_fraction": (
            len(visited_tiles)
            / prepared["valid_tiles"]
        ),
    }


# =============================================================================
# Aggregation
# =============================================================================

def make_run_row(
    workload_meta: Dict,
    prepared: Dict,
    *,
    seed: int,
    popsize: int,
    maxiter: int,
    result: Dict,
) -> Dict:
    ref_m, ref_k, ref_n = result["reference_tile"]
    sel_m, sel_k, sel_n = result["selected_tile"]

    return {
        "workload_label": workload_meta["workload_label"],
        "M": workload_meta["M"],
        "K": workload_meta["K"],
        "N": workload_meta["N"],
        "Data_Type": workload_meta["Data_Type"],
        "valid_tiles": prepared["valid_tiles"],

        "popsize": popsize,
        "maxiter": maxiter,
        "seed": seed,

        "reference_m": ref_m,
        "reference_k": ref_k,
        "reference_n": ref_n,
        "reference_gops": safe_round(
            result["reference_gops"]
        ),

        "selected_m": sel_m,
        "selected_k": sel_k,
        "selected_n": sel_n,
        "selected_gops": safe_round(
            result["selected_gops"]
        ),

        "exact_match": int(result["exact_match"]),
        "gap_pct": safe_round(result["gap_pct"]),
        "near_optimal_1pct": int(
            result["near_optimal_1pct"]
        ),
        "near_optimal_2pct": int(
            result["near_optimal_2pct"]
        ),

        "unique_tiles_evaluated": result[
            "unique_tiles_evaluated"
        ],
        "evaluated_fraction": safe_round(
            result["evaluated_fraction"]
        ),
    }


def summarize_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = [
        "workload_label",
        "M",
        "K",
        "N",
        "Data_Type",
        "valid_tiles",
        "popsize",
        "maxiter",
        "reference_m",
        "reference_k",
        "reference_n",
        "reference_gops",
    ]

    for keys, group in run_df.groupby(
        group_cols,
        sort=False,
        dropna=False,
    ):
        (
            workload_label,
            M,
            K,
            N,
            dtype,
            valid_tiles,
            popsize,
            maxiter,
            reference_m,
            reference_k,
            reference_n,
            reference_gops,
        ) = keys

        gaps = group["gap_pct"].to_numpy(dtype=float)
        unique = group[
            "unique_tiles_evaluated"
        ].to_numpy(dtype=float)
        fractions = group[
            "evaluated_fraction"
        ].to_numpy(dtype=float)

        exact_count = int(group["exact_match"].sum())
        near1_count = int(
            group["near_optimal_1pct"].sum()
        )
        near2_count = int(
            group["near_optimal_2pct"].sum()
        )

        seeds = len(group)

        rows.append(
            {
                "workload_label": workload_label,
                "M": int(M),
                "K": int(K),
                "N": int(N),
                "Data_Type": dtype,
                "valid_tiles": int(valid_tiles),

                "popsize": int(popsize),
                "maxiter": int(maxiter),
                "seeds": seeds,

                "reference_m": int(reference_m),
                "reference_k": int(reference_k),
                "reference_n": int(reference_n),
                "reference_gops": safe_round(
                    reference_gops
                ),

                "exact_match_count": exact_count,
                "exact_match_rate_pct": safe_round(
                    100.0 * exact_count / seeds
                ),

                "near_optimal_1pct_count": near1_count,
                "near_optimal_1pct_rate_pct": safe_round(
                    100.0 * near1_count / seeds
                ),

                "near_optimal_2pct_count": near2_count,
                "near_optimal_2pct_rate_pct": safe_round(
                    100.0 * near2_count / seeds
                ),

                "mean_gap_pct": safe_round(gaps.mean()),
                "median_gap_pct": safe_round(
                    np.median(gaps)
                ),
                "std_gap_pct": safe_round(
                    gaps.std(ddof=0)
                ),
                "worst_gap_pct": safe_round(gaps.max()),

                "avg_unique_tiles_evaluated": safe_round(
                    unique.mean()
                ),
                "min_unique_tiles_evaluated": int(
                    unique.min()
                ),
                "max_unique_tiles_evaluated": int(
                    unique.max()
                ),
                "avg_evaluated_fraction": safe_round(
                    fractions.mean()
                ),
                "avg_evaluated_fraction_pct": safe_round(
                    100.0 * fractions.mean()
                ),
            }
        )

    return pd.DataFrame(rows, columns=SUMMARY_FIELDS)


# =============================================================================
# Replay campaign
# =============================================================================

def run_replay_campaign(
    df: pd.DataFrame,
    *,
    budgets: Sequence[Tuple[int, int]],
    seeds: Iterable[int],
) -> pd.DataFrame:
    rows = []

    grouped = df.groupby(
        WORKLOAD_COLUMNS,
        sort=False,
    )

    for workload_keys, workload_df in grouped:
        (
            label,
            M,
            K,
            N,
            dtype,
        ) = workload_keys

        prepared = prepare_workload(workload_df)

        print("\n" + "=" * 84)
        print(
            f"[Replay] {int(M)}x{int(K)}x{int(N)} ({dtype}) "
            f"[{label}] | valid tiles={prepared['valid_tiles']}"
        )

        ref = prepared["reference_tile"]
        print(
            "[Replay] Exhaustive empirical reference: "
            f"{ref[0]}x{ref[1]}x{ref[2]} "
            f"-> {prepared['reference_gops']:.3f} GOPS"
        )

        if prepared["valid_tiles"] <= EXHAUSTIVE_THRESHOLD:
            print(
                f"[Replay] Skipping Projected-DE: "
                f"{prepared['valid_tiles']} <= "
                f"{EXHAUSTIVE_THRESHOLD}; "
                "real framework uses exhaustive HIL."
            )
            continue

        workload_meta = {
            "workload_label": label,
            "M": int(M),
            "K": int(K),
            "N": int(N),
            "Data_Type": dtype,
        }

        for popsize, maxiter in budgets:
            print(
                f"[Replay] Budget popsize={popsize}, "
                f"maxiter={maxiter} ..."
            )

            for seed in seeds:
                result = replay_one_seed(
                    prepared,
                    seed=int(seed),
                    popsize=int(popsize),
                    maxiter=int(maxiter),
                )

                rows.append(
                    make_run_row(
                        workload_meta,
                        prepared,
                        seed=int(seed),
                        popsize=int(popsize),
                        maxiter=int(maxiter),
                        result=result,
                    )
                )

    return pd.DataFrame(rows, columns=RUN_FIELDS)


# =============================================================================
# CLI / main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay the final Projected-DE search offline using the "
            "exhaustive hardware-measured Chapter 5 dataset."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=(
            "chapter5_data/validation/"
            "chapter5_exhaustive_tiles.csv"
        ),
        help=(
            "Exhaustive tile CSV produced by "
            "collect_chapter5_exhaustive.py."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="chapter5_data/validation",
        help=(
            "Directory for replay CSVs "
            "(default: chapter5_data/validation)."
        ),
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=N_SEEDS,
        help="Number of seeds, starting from 0 (default: 100).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_csv = Path(args.input).expanduser().resolve()
    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input exhaustive CSV not found: {input_csv}"
        )

    df = pd.read_csv(input_csv)
    validate_input(df)

    print(f"[Replay] Input:      {input_csv}")
    print(f"[Replay] Output dir: {output_dir}")
    print(f"[Replay] Rows:       {len(df)}")
    print(
        f"[Replay] Seeds:      0..{args.seeds - 1} "
        f"({args.seeds} total)"
    )
    print(
        "[Replay] Final budget: "
        f"popsize={FINAL_POPSIZE}, maxiter={FINAL_MAXITER}"
    )
    print(
        "[Replay] Parameter sweep: "
        + ", ".join(
            f"{p}x{i}"
            for p, i in PARAMETER_SWEEP
        )
    )

    unique_workloads = df[
        WORKLOAD_COLUMNS
    ].drop_duplicates()

    print(
        f"[Replay] Exhaustive workloads: "
        f"{len(unique_workloads)}"
    )

    # -------------------------------------------------------------------------
    # One single offline campaign for all requested budgets.
    # The final 4x2 rows are later split into chapter5_de_runs/summary.
    # -------------------------------------------------------------------------
    seeds = range(args.seeds)

    all_runs = run_replay_campaign(
        df,
        budgets=PARAMETER_SWEEP,
        seeds=seeds,
    )

    if all_runs.empty:
        raise RuntimeError(
            "No Projected-DE runs were generated."
        )

    sweep_runs_csv = (
        output_dir
        / "chapter5_de_parameter_sweep_runs.csv"
    )
    all_runs.to_csv(
        sweep_runs_csv,
        index=False,
    )

    sweep_summary = summarize_runs(all_runs)

    sweep_summary_csv = (
        output_dir
        / "chapter5_de_parameter_sweep.csv"
    )
    sweep_summary.to_csv(
        sweep_summary_csv,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final deployed 4x2 configuration.
    # -------------------------------------------------------------------------
    final_runs = all_runs[
        (all_runs["popsize"] == FINAL_POPSIZE)
        & (all_runs["maxiter"] == FINAL_MAXITER)
    ].copy()

    final_runs_csv = (
        output_dir
        / "chapter5_de_runs.csv"
    )
    final_runs.to_csv(
        final_runs_csv,
        index=False,
    )

    final_summary = summarize_runs(final_runs)

    final_summary_csv = (
        output_dir
        / "chapter5_de_summary.csv"
    )
    final_summary.to_csv(
        final_summary_csv,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Human-readable terminal summary.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 84)
    print("[Replay] FINAL 4x2 SUMMARY")
    print("=" * 84)

    for _, row in final_summary.iterrows():
        print(
            f"{int(row['M'])}x{int(row['K'])}x{int(row['N'])} "
            f"({row['Data_Type']}): "
            f"exact={row['exact_match_rate_pct']:.1f}% | "
            f"<=1%={row['near_optimal_1pct_rate_pct']:.1f}% | "
            f"<=2%={row['near_optimal_2pct_rate_pct']:.1f}% | "
            f"mean gap={row['mean_gap_pct']:.4f}% | "
            f"worst gap={row['worst_gap_pct']:.4f}% | "
            f"avg unique="
            f"{row['avg_unique_tiles_evaluated']:.2f}/"
            f"{int(row['valid_tiles'])} "
            f"({row['avg_evaluated_fraction_pct']:.2f}%)"
        )

    print("\n" + "=" * 84)
    print("[Replay] Complete. No hardware was used.")
    print(f"[Replay] Final runs:      {final_runs_csv}")
    print(f"[Replay] Final summary:   {final_summary_csv}")
    print(f"[Replay] Sweep summary:   {sweep_summary_csv}")
    print(f"[Replay] All sweep runs:  {sweep_runs_csv}")
    print("=" * 84)


if __name__ == "__main__":
    main()
