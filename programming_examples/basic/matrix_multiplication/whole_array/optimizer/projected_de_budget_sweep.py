import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution


CSV_FILE = "exhaustive_validation_tiles.csv"

N_SEEDS = 100

POPSIZES = [2, 3, 4]
MAXITERS = [1, 2, 3, 5, 8]


def run_projected_de(workload_df, seed, popsize, maxiter):

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
        tuple(tile): gops
        for tile, gops in zip(tile_array, gops_array)
    }

    visited_tiles = set()

    def decode_candidate(x):

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

        return -hardware_results[tile]

    result = differential_evolution(
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

    tuner_tile = decode_candidate(result.x)
    tuner_gops = hardware_results[tuner_tile]

    exhaustive_tile = max(
        hardware_results,
        key=hardware_results.get,
    )

    exhaustive_gops = hardware_results[exhaustive_tile]

    gap = (
        (exhaustive_gops - tuner_gops)
        / exhaustive_gops
        * 100.0
    )

    return {
        "exact": tuner_tile == exhaustive_tile,
        "gap": gap,
        "visited": len(visited_tiles),
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
        total_tiles = len(workload_df)

        print("\n" + "=" * 90)
        print(
            f"WORKLOAD: {M}x{K}x{N} ({dtype}) "
            f"| Valid tiles: {total_tiles}"
        )
        print("=" * 90)

        print(
            f"{'Pop':>4} "
            f"{'Iter':>5} "
            f"{'Exact':>10} "
            f"{'Mean Gap':>12} "
            f"{'Max Gap':>12} "
            f"{'Avg Visited':>14} "
            f"{'Visited %':>12}"
        )

        for popsize in POPSIZES:

            for maxiter in MAXITERS:

                results = []

                for seed in range(N_SEEDS):

                    result = run_projected_de(
                        workload_df,
                        seed,
                        popsize,
                        maxiter,
                    )

                    results.append(result)

                exact = sum(
                    r["exact"] for r in results
                )

                gaps = np.array(
                    [r["gap"] for r in results]
                )

                visited = np.array(
                    [r["visited"] for r in results]
                )

                visited_percent = (
                    visited.mean()
                    / total_tiles
                    * 100.0
                )

                print(
                    f"{popsize:>4} "
                    f"{maxiter:>5} "
                    f"{exact:>7}/{N_SEEDS:<2} "
                    f"{gaps.mean():>11.4f}% "
                    f"{gaps.max():>11.4f}% "
                    f"{visited.mean():>14.2f} "
                    f"{visited_percent:>11.2f}%"
                )


if __name__ == "__main__":
    main()