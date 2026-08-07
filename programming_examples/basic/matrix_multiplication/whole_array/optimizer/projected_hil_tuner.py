import argparse
import numpy as np

from scipy.optimize import differential_evolution

from tiling_optimizer import (
    find_candidates,
    filter_by_memory,
)

from tester import get_hardware_throughput


# ---------------------------------------------------------------------
# Final provisional search parameters derived from exhaustive validation
# ---------------------------------------------------------------------

DE_POPSIZE = 3
DE_MAXITER = 2

# scipy DE population = popsize * number_of_dimensions = 3 * 3 = 9
EXHAUSTIVE_THRESHOLD = 9


def get_valid_configurations(M, K, N, dtype):
    """
    Generate the complete discrete tile space surviving the
    analytical constraints.
    """

    m_candidates, k_candidates, n_candidates, _ = find_candidates(
        M,
        K,
        N,
        dtype=dtype,
        cols=8,
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


def run_hardware_tile(M, K, N, dtype, tile, cache):
    """
    Execute one tile on hardware, using memoization to avoid
    recompiling previously evaluated configurations.
    """

    m, k, n = tile

    if tile in cache:
        print(
            f"   [CACHE] {m}x{k}x{n} -> "
            f"{cache[tile]:.3f} GOPS"
        )
        return cache[tile]

    print(
        f"   [HIL] Evaluating tile "
        f"{m}x{k}x{n}"
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

    cache[tile] = gops

    print(
        f"   [HIL RESULT] "
        f"{m}x{k}x{n} -> {gops:.3f} GOPS"
    )

    return gops


def exhaustive_hil_search(
    M,
    K,
    N,
    dtype,
    valid_configs,
):
    """
    Exhaustively evaluates very small residual design spaces.
    """

    cache = {}

    best_tile = None
    best_gops = -1.0

    for m, k, n, _, _ in valid_configs:

        tile = (m, k, n)

        gops = run_hardware_tile(
            M,
            K,
            N,
            dtype,
            tile,
            cache,
        )

        if gops > best_gops:
            best_gops = gops
            best_tile = tile

    return best_tile, best_gops, cache


def projected_de_hil_search(
    M,
    K,
    N,
    dtype,
    valid_configs,
    seed=42,
):
    """
    Differential Evolution operates in a normalized continuous
    coordinate space. Every continuous candidate is projected onto
    the nearest analytically valid discrete tile configuration.
    """

    tile_array = np.array(
        [
            [m, k, n]
            for m, k, n, _, _ in valid_configs
        ],
        dtype=int,
    )

    # Tile dimensions naturally scale multiplicatively, therefore
    # the projection is performed in logarithmic coordinates.
    log_tiles = np.log2(
        tile_array.astype(float)
    )

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

    cache = {}

    def decode_candidate(x):
        """
        Project continuous DE coordinates onto the nearest
        valid discrete tile.
        """

        x = np.asarray(x)

        distances = np.sum(
            (normalized_tiles - x) ** 2,
            axis=1,
        )

        nearest_index = np.argmin(distances)

        return tuple(
            int(v)
            for v in tile_array[nearest_index]
        )

    def objective_function(x):

        tile = decode_candidate(x)

        gops = run_hardware_tile(
            M,
            K,
            N,
            dtype,
            tile,
            cache,
        )

        return -gops

    result = differential_evolution(
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

    final_tile = decode_candidate(
        result.x
    )

    final_gops = run_hardware_tile(
        M,
        K,
        N,
        dtype,
        final_tile,
        cache,
    )

    # Safety: return the best hardware-measured tile actually
    # encountered during the complete search.
    best_tile = max(
        cache,
        key=cache.get,
    )

    best_gops = cache[best_tile]

    return best_tile, best_gops, cache


def run_super_tuner(
    M,
    K,
    N,
    dtype,
    seed=42,
):

    valid_configs = get_valid_configurations(
        M,
        K,
        N,
        dtype,
    )

    num_configs = len(valid_configs)

    print("\n" + "=" * 72)

    print(
        f"WORKLOAD: "
        f"{M}x{K}x{N} ({dtype})"
    )

    print(
        f"Analytically valid configurations: "
        f"{num_configs}"
    )

    print("=" * 72)

    if num_configs == 0:

        raise RuntimeError(
            "No valid tile configurations found."
        )

    if num_configs <= EXHAUSTIVE_THRESHOLD:

        print(
            "[SEARCH MODE] "
            "Small design space -> Exhaustive HIL"
        )

        best_tile, best_gops, cache = (
            exhaustive_hil_search(
                M,
                K,
                N,
                dtype,
                valid_configs,
            )
        )

        search_mode = "exhaustive"

    else:

        print(
            "[SEARCH MODE] "
            "Projected Differential Evolution HIL"
        )

        print(
            f"[DE] popsize={DE_POPSIZE}, "
            f"maxiter={DE_MAXITER}, "
            f"seed={seed}"
        )

        best_tile, best_gops, cache = (
            projected_de_hil_search(
                M,
                K,
                N,
                dtype,
                valid_configs,
                seed=seed,
            )
        )

        search_mode = "projected_de"

    print("\n" + "-" * 72)

    print(
        f"BEST TILE: "
        f"{best_tile[0]}x"
        f"{best_tile[1]}x"
        f"{best_tile[2]}"
    )

    print(
        f"BEST MEASURED THROUGHPUT: "
        f"{best_gops:.3f} GOPS"
    )

    print(
        f"UNIQUE HARDWARE EVALUATIONS: "
        f"{len(cache)}/{num_configs}"
    )

    print(
        f"SEARCH MODE: {search_mode}"
    )

    print("-" * 72)

    return (
        best_tile,
        best_gops,
        cache,
        search_mode,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--M",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--K",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--N",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--dtype",
        choices=["i16", "bf16"],
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    run_super_tuner(
        args.M,
        args.K,
        args.N,
        args.dtype,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()