import os
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Analytical external-memory traffic model
# =============================================================================

BASELINE_TILE_M = 32
BASELINE_TILE_K = 32
BASELINE_TILE_N = 32

BYTES_PER_INPUT_ELEMENT = 2


def get_output_element_bytes(dtype):
    """
    Return the number of bytes used by one output element.

    BF16 GEMM:
        BF16 output -> 2 bytes.

    INT16 GEMM:
        INT32 output -> 4 bytes.
    """

    if dtype == "bf16":
        return 2

    if dtype == "i16":
        return 4

    raise ValueError(
        f"Unsupported datatype: {dtype}"
    )


def calculate_baseline_memory_traffic(
    M,
    K,
    N,
    dtype,
):


    m = BASELINE_TILE_M
    k = BASELINE_TILE_K
    n = BASELINE_TILE_N

    input_bytes = BYTES_PER_INPUT_ELEMENT
    output_bytes = get_output_element_bytes(
        dtype
    )

    tiles_M = M // m
    tiles_K = K // k
    tiles_N = N // n
    n_aie_rows = 4
    n_aie_cols = 8

    # Matrix A is re-read for every horizontal N tile.
    reads_A = (
        M
        * K
        * input_bytes
        * (N // (n * n_aie_cols))
    )

    # Matrix B is streamed for every M/K/N tile combination.
    reads_B = (
        K
        * N
        * input_bytes
        * (M // (m * n_aie_rows))
    )

    # Matrix C is written once for each output tile.
    writes_C = (
        M
        * N
        * output_bytes
    )

    total = (
        reads_A
        + reads_B
        + writes_C
    )

    return {
        "A_reads": reads_A,
        "B_reads": reads_B,
        "C_writes": writes_C,
        "total": total,
    }


def calculate_optimized_memory_traffic(
    M,
    K,
    N,
    dtype,
    m,
    k,
    n,
):
    """
    Estimate external-memory traffic for the optimized Stationary-A mapping.

    The model assumes that an A tile remains locally resident while the
    corresponding horizontal N tiles are processed. Therefore the A-read
    term does not include the N-tile repetition factor.

    Matrix B remains streamed across the M/K/N traversal, while matrix C
    is written once per output tile.

    This is an analytical estimate of the intended dataflow behavior and
    does not represent a direct measurement of physical DDR transactions.
    """

    input_bytes = BYTES_PER_INPUT_ELEMENT
    output_bytes = get_output_element_bytes(
        dtype
    )

    tiles_M = M // m
    tiles_K = K // k
    tiles_N = N // n
    n_aie_rows = 2
    n_aie_cols = 8

   
    reads_A = (
        M
        * K
        * input_bytes
        * (N // (n * n_aie_cols))
    )

    reads_B = (
        K
        * N
        * input_bytes
        * (M // (m * n_aie_rows))
    )

    # C is written once for each output tile.
    writes_C = (
        M
        * N
        * output_bytes
    )

    total = (
        reads_A
        + reads_B
        + writes_C
    )

    return {
        "A_reads": reads_A,
        "B_reads": reads_B,
        "C_writes": writes_C,
        "total": total,
    }


# =============================================================================
# Unified baseline-vs-optimized visualization
# =============================================================================

def generate_unified_memory_plot(
    M,
    K,
    N,
    dtype,
    opt_m,
    opt_k,
    opt_n,
):
    """
    Generate a grouped bar chart comparing the analytically modeled
    external-memory traffic of the fixed baseline and optimized mapping.

    The plot reports:
        - Matrix A reads;
        - Matrix B reads;
        - Matrix C writes;
        - Descriptor-Derived Transfer Volume.

    The values are analytical estimates derived from the tiling geometry
    and intended DMA dataflow. They are not hardware-counter measurements.
    """

    baseline = (
        calculate_baseline_memory_traffic(
            M,
            K,
            N,
            dtype,
        )
    )

    optimized = (
        calculate_optimized_memory_traffic(
            M,
            K,
            N,
            dtype,
            opt_m,
            opt_k,
            opt_n,
        )
    )

    bytes_per_mb = (
        1024 * 1024
    )

    categories = [
        "Matrix A Reads",
        "Matrix B Reads",
        "Matrix C Writes",
        "Total Traffic",
    ]

    baseline_values = [
        baseline["A_reads"] / bytes_per_mb,
        baseline["B_reads"] / bytes_per_mb,
        baseline["C_writes"] / bytes_per_mb,
        baseline["total"] / bytes_per_mb,
    ]

    optimized_values = [
        optimized["A_reads"] / bytes_per_mb,
        optimized["B_reads"] / bytes_per_mb,
        optimized["C_writes"] / bytes_per_mb,
        optimized["total"] / bytes_per_mb,
    ]

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------

    output_dir = (
        "memory_profiles"
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Plot construction
    # -------------------------------------------------------------------------

    x = np.arange(
        len(categories)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6),
        dpi=150,
    )

    baseline_bars = ax.bar(
        x - width / 2,
        baseline_values,
        width,
        label="Fixed Baseline (32x32x32)",
        color="#e63946",
        edgecolor="black",
    )

    optimized_bars = ax.bar(
        x + width / 2,
        optimized_values,
        width,
        label=(
            "Stationary-A "
            f"({opt_m}x{opt_k}x{opt_n})"
        ),
        color="#2a9d8f",
        edgecolor="black",
    )

    # -------------------------------------------------------------------------
    # Plot formatting
    # -------------------------------------------------------------------------

    ax.set_ylabel(
        "Modeled Data Volume (MiB)",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_title(
        "Descriptor-Derived Transfer Volume\n"
        f"Global Shape: "
        f"{M}x{K}x{N} ({dtype.upper()})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        categories,
        fontsize=10,
    )

    ax.legend(
        fontsize=10,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    # -------------------------------------------------------------------------
    # Numerical labels above bars
    # -------------------------------------------------------------------------

    def add_value_labels(bars):
        for bar in bars:

            height = (
                bar.get_height()
            )

            ax.annotate(
                f"{height:.1f} MiB",
                xy=(
                    bar.get_x()
                    + bar.get_width() / 2,
                    height,
                ),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    add_value_labels(
        baseline_bars
    )

    add_value_labels(
        optimized_bars
    )

    fig.tight_layout()

    # -------------------------------------------------------------------------
    # Save figure
    # -------------------------------------------------------------------------

    output_name = os.path.join(
        output_dir,
        (
            f"unified_memory_profile_"
            f"{M}x{K}x{N}_{dtype}.png"
        ),
    )

    plt.savefig(
        output_name,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    relative_change = (
    optimized["total"]
    / baseline["total"]
    - 1.0
    )

    print(
        f"[Plotter] Modeled baseline traffic: "
        f"{baseline['total'] / bytes_per_mb:.3f} MiB"
    )

    print(
        f"[Plotter] Modeled optimized traffic: "
        f"{optimized['total'] / bytes_per_mb:.3f} MiB"
    )

    print(
    f"[Plotter] Descriptor-derived total-volume change: "
    f"{100.0 * relative_change:+.2f}%"
)

    print(
        f"[Plotter] Image saved to: "
        f"{output_name}"
    )

    return {
        "baseline": baseline,
        "optimized": optimized,
        "relative_volume_change": relative_change,
        "output_file": output_name,
    }