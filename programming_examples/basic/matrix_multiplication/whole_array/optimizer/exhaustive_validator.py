import csv
from scipy.optimize import differential_evolution

from tiling_optimizer import (
    find_candidates,
    filter_by_memory,
    solve_mapping,
)

from tester import get_hardware_throughput


RESULTS_FILE = "exhaustive_validation_results.csv"
TILES_FILE = "exhaustive_validation_tiles.csv"


def get_valid_configurations(M, K, N, dtype):
    """
    Returns the complete set of tile configurations surviving
    the Analytical Constraint Filter.
    """

    m_candidates, k_candidates, n_candidates, _ = find_candidates(
        M, K, N, dtype=dtype, cols=8
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


def run_exhaustive_hardware_search(M, K, N, dtype):
    """
    Executes every analytically valid tile configuration on physical hardware
    and returns the configuration achieving the highest measured throughput.
    """

    valid_configs = get_valid_configurations(M, K, N, dtype)

    print("\n" + "=" * 70)
    print(
        f"[EXHAUSTIVE SEARCH] Workload: "
        f"{M}x{K}x{N} ({dtype})"
    )
    print(f"[EXHAUSTIVE SEARCH] Valid configurations: {len(valid_configs)}")
    print("=" * 70)

    best_tile = None
    best_gops = -1.0

    hardware_results = {}

    for index, (m, k, n, l1_usage, l2_usage) in enumerate(
        valid_configs, start=1
    ):

        print(
            f"\n[EXHAUSTIVE {index}/{len(valid_configs)}] "
            f"Testing tile {m}x{k}x{n}"
        )

        gops = get_hardware_throughput(
            M,
            K,
            N,
            m,
            k,
            n,
            use_poc="1",
            opt_perf="1",
            dtype=dtype,
        )

        tile_key = (m, k, n)
        hardware_results[tile_key] = gops

        print(
            f"   -> {gops:.4f} GOPS | "
            f"L1={l1_usage / 1024:.2f} KB | "
            f"L2={l2_usage / 1024:.2f} KB"
        )

        with open(TILES_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    M,
                    K,
                    N,
                    dtype,
                    m,
                    k,
                    n,
                    l1_usage,
                    l2_usage,
                    gops,
                ]
            )

        if gops > best_gops:
            best_gops = gops
            best_tile = tile_key

    return best_tile, best_gops, hardware_results


def run_super_tuner_from_hardware_table(
    M,
    K,
    N,
    dtype,
    hardware_results,
    seed=42,
):
    """
    Runs the same Differential Evolution search used by the Super-Tuner,
    but reuses the exhaustive hardware measurements as a lookup table.

    This makes it possible to compare the stochastic search against the
    hardware-measured ground truth without recompiling configurations.
    """

    visited_tiles = set()

    def objective_function(params):
        alpha, gamma, sigma = params

        best_config, _ = solve_mapping(
            M,
            K,
            N,
            alpha=alpha,
            gamma=gamma,
            sigma=sigma,
            dtype=dtype,
        )

        if not best_config:
            return 0.0

        m, k, n = best_config[:3]
        tile_key = (m, k, n)

        visited_tiles.add(tile_key)

        gops = hardware_results.get(tile_key, 0.0)

        return -gops

    bounds = [
        (80.0, 250.0),
        (50.0, 200.0),
        (10.0, 50.0),
    ]

    result = differential_evolution(
        objective_function,
        bounds,
        strategy="best1bin",
        maxiter=15,
        popsize=10,
        tol=0.05,
        seed=seed,
    )

    best_alpha, best_gamma, best_sigma = result.x

    best_config, _ = solve_mapping(
        M,
        K,
        N,
        alpha=best_alpha,
        gamma=best_gamma,
        sigma=best_sigma,
        dtype=dtype,
    )

    tuner_tile = tuple(best_config[:3])
    tuner_gops = hardware_results[tuner_tile]

    return (
        tuner_tile,
        tuner_gops,
        len(visited_tiles),
        (best_alpha, best_gamma, best_sigma),
    )


def validate_workload(M, K, N, dtype):
    """
    Complete validation:
    1. Exhaustively benchmark every valid tile.
    2. Establish the hardware-measured optimum.
    3. Run the Super-Tuner search over the same measured landscape.
    4. Compute the optimality gap.
    """

    exhaustive_tile, exhaustive_gops, hardware_results = (
        run_exhaustive_hardware_search(M, K, N, dtype)
    )

    tuner_tile, tuner_gops, visited_tiles, weights = (
        run_super_tuner_from_hardware_table(
            M,
            K,
            N,
            dtype,
            hardware_results,
        )
    )

    if exhaustive_gops > 0:
        optimality_gap = (
            (exhaustive_gops - tuner_gops)
            / exhaustive_gops
            * 100.0
        )
    else:
        optimality_gap = 0.0

    exact_match = tuner_tile == exhaustive_tile

    alpha, gamma, sigma = weights

    print("\n" + "=" * 70)
    print("[VALIDATION RESULT]")
    print(
        f"Workload:             {M}x{K}x{N} ({dtype})"
    )
    print(
        f"Exhaustive Best:      "
        f"{exhaustive_tile[0]}x"
        f"{exhaustive_tile[1]}x"
        f"{exhaustive_tile[2]} "
        f"-> {exhaustive_gops:.4f} GOPS"
    )
    print(
        f"Super-Tuner Best:     "
        f"{tuner_tile[0]}x"
        f"{tuner_tile[1]}x"
        f"{tuner_tile[2]} "
        f"-> {tuner_gops:.4f} GOPS"
    )
    print(
        f"Optimality Gap:       {optimality_gap:.4f}%"
    )
    print(
        f"Exact Tile Match:     {exact_match}"
    )
    print(
        f"Tiles visited by DE:  "
        f"{visited_tiles}/{len(hardware_results)}"
    )
    print(
        f"Best weights:         "
        f"alpha={alpha:.4f}, "
        f"gamma={gamma:.4f}, "
        f"sigma={sigma:.4f}"
    )
    print("=" * 70)

    with open(RESULTS_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                M,
                K,
                N,
                dtype,
                len(hardware_results),
                exhaustive_tile[0],
                exhaustive_tile[1],
                exhaustive_tile[2],
                exhaustive_gops,
                tuner_tile[0],
                tuner_tile[1],
                tuner_tile[2],
                tuner_gops,
                optimality_gap,
                exact_match,
                visited_tiles,
                alpha,
                gamma,
                sigma,
            ]
        )


def initialize_csv_files():

    with open(RESULTS_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "M",
                "K",
                "N",
                "Data_Type",
                "Valid_Configurations",
                "Exhaustive_m",
                "Exhaustive_k",
                "Exhaustive_n",
                "Exhaustive_GOPS",
                "Tuner_m",
                "Tuner_k",
                "Tuner_n",
                "Tuner_GOPS",
                "Optimality_Gap_Percent",
                "Exact_Tile_Match",
                "DE_Visited_Tiles",
                "Alpha",
                "Gamma",
                "Sigma",
            ]
        )

    with open(TILES_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "M",
                "K",
                "N",
                "Data_Type",
                "m",
                "k",
                "n",
                "L1_Bytes",
                "L2_Bytes",
                "Hardware_GOPS",
            ]
        )


def main():

    initialize_csv_files()

    workloads = [
        (512, 512, 512, "i16"),
        (512, 768, 768, "bf16"),
        (512, 512, 2048, "i16"),
    ]

    for M, K, N, dtype in workloads:
        validate_workload(M, K, N, dtype)


if __name__ == "__main__":
    main()