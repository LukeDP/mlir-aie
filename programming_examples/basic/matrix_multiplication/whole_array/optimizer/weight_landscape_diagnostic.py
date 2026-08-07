import random

from tiling_optimizer import solve_mapping


def analyze_workload(M, K, N, dtype, samples=100000):

    selected_tiles = {}

    for _ in range(samples):

        alpha = random.uniform(80.0, 250.0)
        gamma = random.uniform(50.0, 200.0)
        sigma = random.uniform(10.0, 50.0)

        best_config, _ = solve_mapping(
            M,
            K,
            N,
            alpha=alpha,
            gamma=gamma,
            sigma=sigma,
            dtype=dtype,
        )

        if best_config is None:
            continue

        tile = tuple(best_config[:3])

        selected_tiles[tile] = selected_tiles.get(tile, 0) + 1

    print("\n" + "=" * 70)
    print(f"WORKLOAD: {M}x{K}x{N} ({dtype})")
    print(f"Weight samples: {samples}")
    print(f"Unique selected tiles: {len(selected_tiles)}")
    print("=" * 70)

    for tile, count in sorted(
        selected_tiles.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        percentage = 100.0 * count / samples

        print(
            f"{tile[0]}x{tile[1]}x{tile[2]}: "
            f"{count} selections ({percentage:.4f}%)"
        )


def main():

    workloads = [
        (512, 512, 512, "i16"),
        (512, 768, 768, "bf16"),
        (512, 512, 2048, "i16"),
    ]

    for M, K, N, dtype in workloads:
        analyze_workload(M, K, N, dtype)


if __name__ == "__main__":
    main()