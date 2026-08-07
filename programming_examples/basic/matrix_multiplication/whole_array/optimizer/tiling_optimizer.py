# =============================================================================
# Analytical tile-space construction for the AMD Ryzen AI NPU GEMM optimizer
# =============================================================================


# Physical/operational memory limits used by the analytical filter.
#
# Each Compute Tile provides 64 KiB of local data memory. The generated
# core reserves 0xD00 bytes for its stack, leaving the remaining space
# available to the A, B, and C ObjectFIFO buffers.
#
# This bound was validated against the MLIR-AIE basic-sequential allocator:
# configurations whose data buffers exceed this residual capacity fail
# during buffer allocation.
L1_PHYSICAL_LIMIT = 64 * 1024
L1_STACK_RESERVATION = 0xD00
L1_OPERATIONAL_LIMIT = L1_PHYSICAL_LIMIT - L1_STACK_RESERVATION

# Physical L2 capacity of one Memory Tile.
L2_LIMIT = 512 * 1024

# Current whole-array implementation uses two AIE rows and eight columns.
DEFAULT_ROWS = 2
DEFAULT_COLS = 8


# =============================================================================
# Utility functions
# =============================================================================

def get_divisors(n):
    """
    Return all positive integer divisors of n.
    """
    return [
        i
        for i in range(1, n + 1)
        if n % i == 0
    ]


def get_microkernel_dimensions(dtype):
    """
    Return the native vectorized microkernel blocking factors (r, s, t).

    The factors correspond to the M, K, and N blocking dimensions
    used by the evaluated XDNA2 vectorized GEMM kernels.

    BF16:
        4 x 8 x 8

    INT16:
        4 x 4 x 8
    """

    if dtype == "bf16":
        return 4, 8, 8

    if dtype == "i16":
        return 4, 4, 8

    raise ValueError(
        f"Unsupported datatype for tile optimization: {dtype}"
    )


# =============================================================================
# Candidate generation
# =============================================================================

def find_candidates(
    M,
    K,
    N,
    dtype="bf16",
    cols=DEFAULT_COLS,
):
    """
    Generate geometrically valid tile dimensions before memory filtering.

    Candidate generation applies:

      1. global-shape divisibility;
      2. spatial-array divisibility;
      3. vectorized microkernel compatibility.

    In particular, the evaluated vectorized kernels require:

        m % (2 * r) == 0

    rather than only:

        m % r == 0

    For both BF16 and INT16, r = 4, therefore m must be a multiple
    of 8 in the current implementation.
    """

    r, s, t = get_microkernel_dimensions(
        dtype
    )

    rows = DEFAULT_ROWS

    # -----------------------------------------------------------------
    # M dimension
    #
    # m must:
    #   - divide the global M dimension;
    #   - satisfy the vectorized-kernel requirement m % (2*r) == 0;
    #   - allow the global M dimension to be distributed across
    #     the two active AIE rows.
    # -----------------------------------------------------------------

    m_candidates = [
        m
        for m in get_divisors(M)
        if (
            m % (2 * r) == 0
            and M % (m * rows) == 0
        )
    ]

    # -----------------------------------------------------------------
    # K dimension
    #
    # k must divide the global reduction dimension and be aligned
    # with the native microkernel K blocking factor.
    # -----------------------------------------------------------------

    k_candidates = [
        k
        for k in get_divisors(K)
        if k % s == 0
    ]

    # -----------------------------------------------------------------
    # N dimension
    #
    # n must:
    #   - divide the global N dimension;
    #   - satisfy the current vectorized horizontal alignment;
    #   - allow the global N dimension to be distributed across
    #     the active AIE columns.
    # -----------------------------------------------------------------

    n_candidates = [
        n
        for n in get_divisors(N)
        if (
            n % (2 * t) == 0
            and N % (n * cols) == 0
        )
    ]

    return (
        m_candidates,
        k_candidates,
        n_candidates,
        rows,
    )


# =============================================================================
# Analytical constraint filter
# =============================================================================

def filter_by_memory(
    m_list,
    k_list,
    n_list,
    M_global,
    K_global,
    N_global,
    dtype="bf16",
    cols=DEFAULT_COLS,
):
    """
    Filter candidate tile combinations using the analytical constraints.

    The filter evaluates:

      1. DMA-transfer/allocation consistency;
      2. conservative L1 memory usage;
      3. L2 Memory Tile capacity.

    The function returns tuples:

        (m, k, n, l1_usage, l2_usage)
    """

    valid_combinations = []

    rows = DEFAULT_ROWS

    # Number of A tiles assigned to each shim path.
    #
    # For the current 8-column configuration this evaluates to 1.
    n_A_tiles_per_shim = (
        max(1, rows // cols)
        if cols < 4
        else 1
    )

    # Input operands A and B use 16-bit elements.
    input_bytes = 2

    # BF16 accumulates/stores BF16 output in the evaluated path,
    # whereas INT16 GEMM uses INT32 output elements.
    c_bytes = (
        2
        if dtype == "bf16"
        else 4
    )

    for m in m_list:
        for k in k_list:
            for n in n_list:

                # =====================================================
                # 1. DMA-transfer/allocation consistency
                # =====================================================

                iterations_N = (
                    N_global
                    // n
                    // cols
                )

                iterations_K = (
                    K_global
                    // k
                )

                # Logical transfer volume described by the current
                # Stationary-A DMA geometry.
                total_transfer_elements = (
                    iterations_N
                    * iterations_K
                    * (
                        m
                        * n_A_tiles_per_shim
                    )
                    * k
                )

                # A row tile is allocated in L2 across the complete
                # global K dimension.
                allocated_elements = (
                    m
                    * K_global
                )

                if (
                    total_transfer_elements
                    > allocated_elements
                ):
                    continue

                # =====================================================
                # 2. Compute-Tile L1 memory model
                # =====================================================
                #
                # Double buffering is represented by the leading
                # factor of 2.
                #
                # A: BF16/INT16 input -> 2 bytes
                # B: BF16/INT16 input -> 2 bytes
                # C: output-buffer element size depends on the evaluated datatype:
                #    BF16 -> 2 bytes, INT16 GEMM -> INT32 output -> 4 bytes
                # =====================================================

                l1_usage = 2 * (
                    (m * k * input_bytes)
                    + (k * n * input_bytes)
                    + (m * n * c_bytes)
                )

                # =====================================================
                # 3. Memory-Tile L2 model
                # =====================================================
                #
                # A:
                #   depth-2 full-K buffering of one m x K_global slice                
                # B:
                #   double-buffered k x n tile
                #
                # C:
                #   double-buffered output storage, with the additional
                #   factor matching the current whole-array allocation
                #   scheme.
                # =====================================================

                l2_usage = (
                    2 * (m * K_global * 2)
                    + 2 * (k * n * 2)
                    + 2 * (m * n * c_bytes * 2)
                )

                # =====================================================
                # 4. Capacity and empirical compiler-safety constraints
                # =====================================================

                if (
                    l1_usage
                    <= L1_OPERATIONAL_LIMIT
                    and l2_usage
                    <= L2_LIMIT
                ):


                    valid_combinations.append(
                        (
                            m,
                            k,
                            n,
                            l1_usage,
                            l2_usage,
                        )
                    )

    return valid_combinations


# =============================================================================
# Legacy analytical ranking model
# =============================================================================
#
# IMPORTANT:
#
# The functions below are retained for backward compatibility and for
# analytical diagnostics.
#
# They are NOT used by the final Hardware-in-the-Loop Super-Tuner in
# main_launcher_wBenchmark.py.
#
# The final tuner directly explores the discrete tile space surviving
# find_candidates() and filter_by_memory().
#
# solve_mapping() is kept because whole_array_poc_heuristic.py still
# uses it as a fallback when explicit tile dimensions are not supplied.
# =============================================================================

def calculate_score(
    m,
    k,
    n,
    M,
    K,
    N,
    alpha,
    gamma,
    sigma,
    dtype="bf16",
):
    """
    Compute the legacy analytical tile-ranking score.

    This model is retained for backward compatibility and diagnostic
    comparison with the previous weight-space optimization method.

    It is not the fitness function of the final Projected-DE HIL tuner.
    """

    tile_ops = (
        2
        * m
        * n
        * k
    )

    c_bytes = (
        2
        if dtype == "bf16"
        else 4
    )

    # -----------------------------------------------------------------
    # Analytical arithmetic-intensity proxy
    # -----------------------------------------------------------------

    tile_bytes = (
        (m * k * 2)
        + (k * n * 2)
        + (
            m
            * n
            * c_bytes
            / (K / k)
        )
    )

    tile_ai = (
        tile_ops
        / tile_bytes
    )

    # -----------------------------------------------------------------
    # Legacy NPU-efficiency proxy
    # -----------------------------------------------------------------

    k_div = (
        64
        if dtype == "bf16"
        else 32
    )

    # Heuristic penalty retained from the previous analytical model.
    # It approximates the increased cost associated with very small
    # horizontal tiles in the evaluated 8-column mapping.
    n_cols = DEFAULT_COLS

    interconnect_contention_factor = (
        n_cols ** 2
        if n < 64
        else 1.0
    )

    term_npu = (
        (128 / m) ** 3
        + (k_div / k) ** 2
        + (
            (64 / n) ** 2
            * interconnect_contention_factor
        )
    )

    # -----------------------------------------------------------------
    # Legacy bandwidth proxy
    # -----------------------------------------------------------------

    term_bandwidth = (
        1
        / tile_ai
    )

    # -----------------------------------------------------------------
    # Legacy host/setup proxy
    # -----------------------------------------------------------------

    num_iterations = (
        (M // m)
        * (
            N
            // (
                n
                * DEFAULT_COLS
            )
        )
    )

    term_setup = (
        num_iterations
        * 1.2
    )

    # -----------------------------------------------------------------
    # Weighted analytical objective
    # -----------------------------------------------------------------

    j = (
        alpha
        * term_bandwidth
        + gamma
        * term_npu
        + sigma
        * term_setup
    )

    return (
        j,
        tile_ai,
    )


def solve_mapping(
    M,
    K,
    N,
    alpha=150.0,
    gamma=100.0,
    sigma=30.0,
    dtype="bf16",
):
    """
    Legacy deterministic analytical tile selector.

    The function evaluates every configuration surviving the analytical
    candidate-generation and filtering stages and returns the tile with
    the minimum legacy analytical score.

    This function is retained for:
      - backward compatibility with whole_array_poc_heuristic.py;
      - diagnostic comparison with the previous optimization method.

    The final Super-Tuner does not use this function for HIL search.
    """

    (
        m_candidates,
        k_candidates,
        n_candidates,
        _,
    ) = find_candidates(
        M,
        K,
        N,
        dtype=dtype,
        cols=DEFAULT_COLS,
    )

    valid_configs = filter_by_memory(
        m_candidates,
        k_candidates,
        n_candidates,
        M,
        K,
        N,
        dtype=dtype,
        cols=DEFAULT_COLS,
    )

    best_config = None
    min_j = float("inf")

    results = []

    for (
        m,
        k,
        n,
        _l1_usage,
        _l2_usage,
    ) in valid_configs:

        j, tile_ai = calculate_score(
            m,
            k,
            n,
            M,
            K,
            N,
            alpha,
            gamma,
            sigma,
            dtype=dtype,
        )

        results.append(
            (
                m,
                k,
                n,
                j,
                tile_ai,
            )
        )

        if j < min_j:
            min_j = j

            best_config = (
                m,
                k,
                n,
                j,
                tile_ai,
            )

    return (
        best_config,
        results,
    )