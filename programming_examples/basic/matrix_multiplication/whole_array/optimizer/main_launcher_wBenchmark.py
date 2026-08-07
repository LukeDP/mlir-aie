import argparse
import subprocess
import json
import os
import sys
import numpy as np

from scipy.optimize import differential_evolution
from memory_plotter import generate_unified_memory_plot

from tiling_optimizer import (
    find_candidates,
    filter_by_memory,
)


# =============================================================================
# Persistent cache and search configuration
# =============================================================================

# The previous implementation cached Differential Evolution weights.
# The new implementation stores the best hardware-measured tile directly.
CACHE_FILE = "best_tiles_cache.json"

# Projected Differential Evolution parameters.
# These values were selected through the offline exhaustive-validation sweep.
DE_POPSIZE = 4
DE_MAXITER = 2
DE_SEED = 42

# If the residual analytical design space is sufficiently small,
# exhaustive HIL evaluation is cheaper and guarantees the best measured tile.
EXHAUSTIVE_THRESHOLD = 12

# Number of hardware repetitions used during stochastic search.
TUNING_ITERATIONS = 15

# Number of hardware repetitions used for the final reported benchmark.
FINAL_ITERATIONS = 20


# =============================================================================
# Persistent cache management
# =============================================================================

def load_cache():
    """
    Load the persistent tile cache.

    The cache contains:
      - best tile results for previously optimized workloads;
      - the last compiled datatype;
      - the last execution mode.

    If the file does not exist or is corrupted, an empty cache is returned.
    """
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            try:
                return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}

    return {}


def save_cache(cache):
    """Persist the current cache to disk."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)


# =============================================================================
# Build-state management
# =============================================================================

def clean_build_artifacts():
    """
    Remove build artifacts that may become stale when changing
    datatype or execution mode.

    The MLIR-AIE/Peano build flow may reuse previously generated
    objects, therefore a state transition requires a clean rebuild.
    """
    try:
        if os.path.exists("../whole_array.exe"):
            os.remove("../whole_array.exe")

        # Keep compatibility with both build-directory layouts used
        # during development of the project.
        for build_dir in ("../_build", "../build"):
            if os.path.exists(build_dir):
                subprocess.run(
                    ["rm", "-rf", build_dir],
                    check=True,
                )

    except Exception as exc:
        print(f"[Warning] Build cleanup failed: {exc}")


def ensure_build_state(cache, dtype, mode):
    """
    Ensure that the build state matches the requested datatype
    and execution mode.

    Build artifacts are removed only when either the datatype or
    the execution mode changes.
    """
    last_dtype = cache.get("last_compiled_dtype", "")
    last_mode = cache.get("last_compiled_mode", "")

    if last_dtype != dtype or last_mode != mode:
        print(
            "[Launcher] Configuration change detected "
            f"(dtype: {last_dtype or 'none'} -> {dtype}, "
            f"mode: {last_mode or 'none'} -> {mode}). "
            "Cleaning build artifacts..."
        )

        clean_build_artifacts()

        cache["last_compiled_dtype"] = dtype
        cache["last_compiled_mode"] = mode

        save_cache(cache)

    return cache


# =============================================================================
# Hardware-output parsing
# =============================================================================

def parse_average_throughput(stdout):
    """
    Extract the final average throughput reported by the Makefile
    benchmark pipeline.

    Returns 0.0 if the expected output marker cannot be parsed.
    """
    for line in stdout.splitlines():
        if "Final Average Throughput:" in line:
            try:
                return float(
                    line.split(":")[1].split()[0]
                )
            except (ValueError, IndexError):
                return 0.0

    return 0.0


# =============================================================================
# Hardware-in-the-Loop tile evaluation
# =============================================================================

def run_tuning_hardware_test(
    M,
    K,
    N,
    m,
    k,
    n,
    dtype,
    cache,
):
    """
    Compile and execute one optimized tile during the HIL search.

    The returned value is the average hardware throughput in GOPS.
    A failed compilation or execution is assigned 0.0 GOPS so that
    the stochastic optimizer cannot prefer the failed configuration.
    """

    ensure_build_state(
        cache,
        dtype,
        "optimized",
    )

    clean_build_artifacts()

    dtype_out = (
        "bf16"
        if dtype == "bf16"
        else "i32"
    )

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

    result = subprocess.run(
        cmd,
        cwd="..",
        capture_output=True,
        text=True,
    )

    # A non-zero return code indicates a compilation or runtime failure.
    if result.returncode != 0:
        print("FAILED")

        diagnostic_output = (
            result.stderr
            if result.stderr
            else result.stdout
        )

        if diagnostic_output:
            print(
                diagnostic_output[-2000:]
            )

        return 0.0

    gops = parse_average_throughput(
        result.stdout
    )

    if gops <= 0.0:
        print("NO VALID THROUGHPUT")
        return 0.0

    print(
        f"{gops:.3f} GOPS"
    )

    return gops


# =============================================================================
# Analytical candidate-space construction
# =============================================================================

def get_valid_configurations(
    M,
    K,
    N,
    dtype,
):
    """
    Generate the complete discrete tile space surviving the
    analytical candidate-generation and filtering stages.

    Candidate generation enforces the microkernel/geometric
    constraints, while filter_by_memory applies the DMA-transfer,
    L1/L2-memory, and workload-dependent analytical rules.
    """

    m_candidates, k_candidates, n_candidates, _ = (
        find_candidates(
            M,
            K,
            N,
            dtype=dtype,
            cols=8,
        )
    )

    valid_configs = filter_by_memory(
        m_candidates,
        k_candidates,
        n_candidates,
        M,
        K,
        N,
        dtype=dtype,
        cols=8,
    )

    return valid_configs


# =============================================================================
# Hardware-in-the-Loop Super-Tuner
# =============================================================================

def run_super_tuner(
    M,
    K,
    N,
    dtype,
    seed=DE_SEED,
):
    """
    Run the Hardware-in-the-Loop tile-space Super-Tuner.

    The optimization procedure consists of two stages:

      1. The analytical filter constructs the set of admissible
         discrete tile configurations.

      2. The residual design space is explored either:
           - exhaustively, when the number of valid configurations
             is small;
           - through Projected Differential Evolution, when the
             residual search space is larger.

    Differential Evolution operates in a normalized continuous
    coordinate space. Every continuous point is projected onto
    the nearest analytically valid discrete tile in log2 space.

    The physical NPU throughput is the optimization fitness.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        f"[SUPER-TUNER] Tile-space search for "
        f"{M}x{K}x{N} ({dtype})"
    )

    print(
        "=" * 60
    )

    # -----------------------------------------------------------------
    # Step 1: construct the analytically valid discrete design space.
    # -----------------------------------------------------------------

    valid_configs = get_valid_configurations(
        M,
        K,
        N,
        dtype,
    )

    if not valid_configs:
        raise RuntimeError(
            f"No valid tile configurations found for "
            f"{M}x{K}x{N} ({dtype})."
        )

    tile_array = np.array(
        [
            [m, k, n]
            for m, k, n, _, _ in valid_configs
        ],
        dtype=int,
    )

    num_valid_configs = len(
        tile_array
    )

    print(
        f"[SUPER-TUNER] Analytically valid configurations: "
        f"{num_valid_configs}"
    )

    # -----------------------------------------------------------------
    # Intra-session hardware memoization.
    #
    # Key:
    #     (m, k, n)
    #
    # Value:
    #     measured throughput in GOPS
    # -----------------------------------------------------------------

    tile_benchmark_cache = {}

    # Persistent cache is kept separate from the intra-session
    # hardware-measurement dictionary.
    persistent_cache = load_cache()

    def evaluate_tile(tile):
        """
        Evaluate one discrete tile on physical hardware.

        If the same tile is proposed again by Differential Evolution,
        the previously measured throughput is returned immediately.
        """

        tile = tuple(
            int(value)
            for value in tile
        )

        # -------------------------------------------------------------
        # Intra-session cache hit.
        # -------------------------------------------------------------

        if tile in tile_benchmark_cache:
            cached_gops = (
                tile_benchmark_cache[tile]
            )

            print(
                f"  > Cache hit: "
                f"{tile[0]}x"
                f"{tile[1]}x"
                f"{tile[2]} "
                f"-> {cached_gops:.3f} GOPS"
            )

            return cached_gops

        # -------------------------------------------------------------
        # New physical hardware evaluation.
        # -------------------------------------------------------------

        m, k, n = tile

        tile_key = (
            f"{m}x{k}x{n}"
        )

        print(
            f"  > Hardware Test: {tile_key}",
            end="... ",
            flush=True,
        )

        gops = run_tuning_hardware_test(
            M,
            K,
            N,
            m,
            k,
            n,
            dtype,
            persistent_cache,
        )

        tile_benchmark_cache[tile] = gops

        return gops

    # =========================================================================
    # Search mode A:
    # Exhaustive HIL search for very small residual spaces.
    # =========================================================================

    if (
        num_valid_configs
        <= EXHAUSTIVE_THRESHOLD
    ):

        print(
            f"[SUPER-TUNER] Search mode: "
            f"Exhaustive HIL "
            f"({num_valid_configs} <= "
            f"{EXHAUSTIVE_THRESHOLD})"
        )

        for tile in tile_array:
            evaluate_tile(tile)

        search_mode = "exhaustive"

    # =========================================================================
    # Search mode B:
    # Projected Differential Evolution for larger spaces.
    # =========================================================================

    else:

        print(
            f"[SUPER-TUNER] Search mode: "
            f"Projected DE HIL "
            f"(popsize={DE_POPSIZE}, "
            f"maxiter={DE_MAXITER}, "
            f"seed={seed})"
        )

        # -------------------------------------------------------------
        # Transform discrete tile dimensions into log2 coordinates.
        #
        # Tile dimensions naturally vary multiplicatively:
        #     8, 16, 32, 64, ...
        #
        # Therefore log2 coordinates provide a more meaningful
        # geometric representation than raw Euclidean coordinates.
        # -------------------------------------------------------------

        log_tiles = np.log2(
            tile_array.astype(float)
        )

        coord_min = log_tiles.min(
            axis=0
        )

        coord_max = log_tiles.max(
            axis=0
        )

        # Prevent division by zero when one tile dimension is constant
        # throughout the complete residual design space.
        coord_range = np.where(
            coord_max > coord_min,
            coord_max - coord_min,
            1.0,
        )

        normalized_tiles = (
            log_tiles - coord_min
        ) / coord_range

        def decode_candidate(x):
            """
            Project a continuous Differential Evolution point onto
            the nearest valid discrete tile.

            Distance is measured in normalized log2 tile space.
            """

            x = np.asarray(x)

            distances = np.sum(
                (
                    normalized_tiles
                    - x
                ) ** 2,
                axis=1,
            )

            nearest_index = np.argmin(
                distances
            )

            tile = tuple(
                int(value)
                for value
                in tile_array[
                    nearest_index
                ]
            )

            return tile

        def objective_function(x):
            """
            Differential Evolution fitness function.

            scipy.optimize.differential_evolution performs
            minimization, therefore measured hardware throughput
            is negated.
            """

            tile = decode_candidate(x)

            gops = evaluate_tile(tile)

            return -gops

        # -------------------------------------------------------------
        # Run Projected Differential Evolution.
        #
        # popsize = 3
        # optimized dimensions = 3
        #
        # Nominal initial population:
        #     3 * 3 = 9 individuals
        #
        # maxiter = 2 was selected from the exhaustive-validation
        # budget sweep as a compromise between HIL cost and
        # near-optimal hardware performance.
        # -------------------------------------------------------------

        differential_evolution(
            objective_function,
            bounds=[
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
            ],
            strategy="best1bin",
            popsize=DE_POPSIZE,
            maxiter=DE_MAXITER,
            tol=0.0,
            atol=0.0,
            seed=seed,
            polish=False,
        )

        search_mode = (
            "projected_de"
        )

    # =========================================================================
    # Select the best hardware-measured tile encountered during search.
    # =========================================================================

    if not tile_benchmark_cache:
        raise RuntimeError(
            "The Super-Tuner completed without "
            "evaluating any tile."
        )

    best_tile = max(
        tile_benchmark_cache,
        key=tile_benchmark_cache.get,
    )

    best_gops = (
        tile_benchmark_cache[
            best_tile
        ]
    )

    # If every configuration failed, do not silently return an
    # arbitrary zero-throughput tile.
    if best_gops <= 0.0:
        raise RuntimeError(
            "All HIL evaluations failed or "
            "returned zero throughput."
        )

    print(
        f"[SUPER-TUNER] Best measured tile: "
        f"{best_tile[0]}x"
        f"{best_tile[1]}x"
        f"{best_tile[2]} "
        f"-> {best_gops:.3f} GOPS"
    )

    print(
        f"[SUPER-TUNER] Unique HIL evaluations: "
        f"{len(tile_benchmark_cache)}/"
        f"{num_valid_configs}"
    )

    # Store the discrete tile itself rather than the obsolete
    # analytical weight vector.
    return {
        "m": int(best_tile[0]),
        "k": int(best_tile[1]),
        "n": int(best_tile[2]),
        "gops": float(best_gops),
        "search_mode": search_mode,
        "evaluated_tiles": int(
            len(tile_benchmark_cache)
        ),
        "valid_tiles": int(
            num_valid_configs
        ),
        "seed": int(seed),
    }


# =============================================================================
# Final benchmark execution
# =============================================================================

def run_test(
    M,
    K,
    N,
    tag,
    mode,
    cache,
    dtype,
):
    """
    Run one final hardware benchmark.

    Baseline mode:
        fixed 32x32x32 tile.

    Optimized mode:
        use the best tile returned by the HIL Super-Tuner or retrieve
        a previously optimized tile from the persistent cache.
    """

    print(
        f"\nTEST {tag}: "
        f"{M}x{K}x{N} "
        f"({mode}) - "
        f"Type: {dtype}"
    )

    # =========================================================================
    # Optimized execution
    # =========================================================================

    if mode == "optimized":

        shape_key = (
            f"{M}x{K}x{N}_{dtype}"
        )

        # -------------------------------------------------------------
        # Run the tuner only if this workload has never been optimized.
        # -------------------------------------------------------------

        if shape_key not in cache:

            best_result = run_super_tuner(
                M,
                K,
                N,
                dtype,
                seed=DE_SEED,
            )

            # Reload the persistent cache because run_super_tuner()
            # may have updated build-state metadata.
            cache = load_cache()

            cache[shape_key] = (
                best_result
            )

            save_cache(cache)

        # -------------------------------------------------------------
        # Retrieve the cached best discrete tile directly.
        # No analytical weight-to-tile reconstruction is required.
        # -------------------------------------------------------------

        best_result = (
            cache[shape_key]
        )

        m = int(
            best_result["m"]
        )

        k = int(
            best_result["k"]
        )

        n = int(
            best_result["n"]
        )

        use_poc = "1"
        opt_perf = "1"

        print(
            f"[Launcher] Using optimized tile "
            f"{m}x{k}x{n}."
        )

    # =========================================================================
    # Baseline execution
    # =========================================================================

    else:

        m = 32
        k = 32
        n = 32

        use_poc = "0"
        opt_perf = "0"

        print(
            "[Launcher] Using fixed baseline tile "
            "32x32x32."
        )

    # -----------------------------------------------------------------
    # Ensure that stale artifacts from another mode/datatype are removed.
    # -----------------------------------------------------------------

    ensure_build_state(
        cache,
        dtype,
        mode,
    )

    dtype_out = (
        "bf16"
        if dtype == "bf16"
        else "i32"
    )

    # -----------------------------------------------------------------
    # Final benchmark.
    #
    # This execution uses more iterations than the intermediate HIL
    # evaluations because it is intended for the final reported result.
    # -----------------------------------------------------------------

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
        f"ITERATIONS={FINAL_ITERATIONS}",
        f"opt_perf={opt_perf}",
        f"dtype_in={dtype}",
        f"dtype_out={dtype_out}",
    ]

    try:
        subprocess.run(
            cmd,
            cwd="..",
            check=True,
        )

    except subprocess.CalledProcessError:

        print(
            f"[Error] Execution failed for "
            f"configuration {tag}."
        )

        sys.exit(1)

    return cache


# =============================================================================
# Main launcher
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Performance launcher for "
            "AIE matrix multiplication."
        )
    )

    parser.add_argument(
        "-b",
        "--baseline",
        action="store_true",
        help=(
            "Run baseline benchmarks."
        ),
    )

    parser.add_argument(
        "-o",
        "--optimized",
        action="store_true",
        help=(
            "Run optimized benchmarks "
            "with the Super-Tuner."
        ),
    )

    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=[
            "i16",
            "bf16",
        ],
        default="bf16",
        help=(
            "Input datatype: "
            "i16 (integer) or "
            "bf16 (Brain Floating Point)."
        ),
    )

    parser.add_argument(
        "-p",
        "--plot-memory",
        action="store_true",
        help=(
            "Generate analytical "
            "memory-access charts."
        ),
    )

    args = parser.parse_args()

    # Require an explicit and unambiguous execution mode.
    if (
        args.baseline
        == args.optimized
    ):
        parser.error(
            "Select exactly one execution mode: "
            "--baseline or --optimized."
        )

    mode = (
        "optimized"
        if args.optimized
        else "baseline"
    )

    cache = load_cache()

    # -----------------------------------------------------------------
    # Representative evaluation workloads.
    # -----------------------------------------------------------------

    tests = [
        (
            512,
            512,
            512,
            "BALANCED",
        ),
        (
            512,
            768,
            768,
            "BERT_SHAPE",
        ),
        (
            512,
            512,
            2048,
            "MEMORY_STRESS_N",
        ),
        (
            512,
            2048,
            512,
            "REDUCTION_STRESS_K",
        ),
    ]

    for M, K, N, tag in tests:

        # run_test() returns the updated cache so that newly discovered
        # optimized tiles are immediately visible to subsequent steps.
        cache = run_test(
            M,
            K,
            N,
            tag,
            mode,
            cache,
            dtype=args.type,
        )

        # =====================================================================
        # Optional analytical memory-traffic visualization
        # =====================================================================

        if args.plot_memory:

            # The unified memory profile compares the fixed baseline
            # against the optimized Stationary-A mapping. Therefore an
            # optimized tile must be available before generating it.
            if mode == "optimized":

                shape_key = (
                    f"{M}x{K}x{N}_"
                    f"{args.type}"
                )

                best_result = cache.get(
                    shape_key
                )

                if best_result:

                    opt_m = int(
                        best_result["m"]
                    )

                    opt_k = int(
                        best_result["k"]
                    )

                    opt_n = int(
                        best_result["n"]
                    )

                    generate_unified_memory_plot(
                        M,
                        K,
                        N,
                        args.type,
                        opt_m,
                        opt_k,
                        opt_n,
                    )

                else:
                    print(
                        "[Plotter] No optimized tile is available "
                        "for this workload."
                    )

            else:
                print(
                    "[Plotter] Unified memory comparison is generated "
                    "only in optimized mode."
                )