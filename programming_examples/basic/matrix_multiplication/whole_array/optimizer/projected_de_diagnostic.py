import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution


CSV_FILE = "exhaustive_validation_tiles.csv"
N_SEEDS = 20


def run_projected_de(workload_df, seed):

    tile_array = workload_df[["m", "k", "n"]].to_numpy(dtype=int)
    gops_array = workload_df["Hardware_GOPS"].to_numpy(dtype=float)

    # Logarithmic coordinates are used because tile dimensions
    # naturally vary multiplicatively.
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
        tuple(tile): gops
        for tile, gops in zip(tile_array, gops_array)
    }

    visited_tiles = set()

    def decode_candidate(x):
        """
        Projects a continuous DE point onto the nearest
        analytically valid discrete tile.
        """

        x = np.asarray(x)

        distances = np.sum(
            (normalized_tiles - x) ** 2,
            axis=1,
        )

        nearest_index = np.argmin(distances)

        return tuple(tile_array[nearest_index])

    def objective_function(x):

        tile = decode_candidate(x)

        visited_tiles.add(tile)

        gops = hardware_results[tile]

        return -gops

    result = differential_evolution(
        objective_function,
        bounds=[
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
        ],
        strategy="best1bin",
        maxiter=15,
        popsize=10,
        tol=0.05,
        seed=seed,
        polish=False,
    )

    tuner_tile = decode_candidate(result.x)
    tuner_gops = hardware_results[tuner_tile]

    exhaustive_tile = max(
        hardware_results,
        key=hardware_results.get,
    )

    exhaustive_gops = hardware_results[exhaustive_tile]

    optimality_gap = (
        (exhaustive_gops - tuner_gops)
        / exhaustive_gops
        * 100.0
    )

    return {
        "seed": seed,
        "tuner_tile": tuner_tile,
        "tuner_gops": tuner_gops,
        "exhaustive_tile": exhaustive_tile,
        "exhaustive_gops": exhaustive_gops,
        "gap": optimality_gap,
        "visited_tiles": len(visited_tiles),
        "valid_tiles": len(hardware_results),
        "exact_match": tuner_tile == exhaustive_tile,
    }


def main():

    df = pd.read_csv(CSV_FILE)

    workload_columns = [
        "M",
        "K",
        "N",
        "Data_Type",
    ]

    for workload, workload_df in df.groupby(workload_columns):

        M, K, N, dtype = workload

        print("\n" + "=" * 72)
        print(
            f"WORKLOAD: {M}x{K}x{N} ({dtype})"
        )
        print(
            f"Valid configurations: {len(workload_df)}"
        )
        print("=" * 72)

        results = []

        for seed in range(N_SEEDS):

            result = run_projected_de(
                workload_df,
                seed,
            )

            results.append(result)

            print(
                f"Seed {seed:02d}: "
                f"{result['tuner_tile']} -> "
                f"{result['tuner_gops']:.3f} GOPS | "
                f"Gap={result['gap']:.4f}% | "
                f"Visited="
                f"{result['visited_tiles']}/"
                f"{result['valid_tiles']}"
            )

        exact_matches = sum(
            result["exact_match"]
            for result in results
        )

        gaps = np.array(
            [result["gap"] for result in results]
        )

        visited = np.array(
            [
                result["visited_tiles"]
                for result in results
            ]
        )

        exhaustive_tile = results[0]["exhaustive_tile"]
        exhaustive_gops = results[0]["exhaustive_gops"]

        print("\nSUMMARY")
        print(
            f"Exhaustive optimum: "
            f"{exhaustive_tile} -> "
            f"{exhaustive_gops:.3f} GOPS"
        )
        print(
            f"Exact optimum recovery: "
            f"{exact_matches}/{N_SEEDS}"
        )
        print(
            f"Mean optimality gap: "
            f"{gaps.mean():.4f}%"
        )
        print(
            f"Maximum optimality gap: "
            f"{gaps.max():.4f}%"
        )
        print(
            f"Average unique tiles visited: "
            f"{visited.mean():.2f}"
        )


if __name__ == "__main__":
    main()