import os
import json
import csv
import subprocess
from scipy.optimize import differential_evolution
from tiling_optimizer import solve_mapping, find_candidates

# Output database configuration
CSV_FILE = "experimental_dataset_massive.csv"

def calculate_analytical_bytes(M, K, N, m, k, n, dtype, mode):
    """
    Analytically calculates the total volume of data transferred to/from DDR memory.
    Derived from the physical ObjectFIFO behavior inside memory_plotter.py.
    """
    in_bytes = 2
    c_bytes = 2 if dtype == "bf16" else 4
    
    tiles_M = M // m
    tiles_N = N // n
    tiles_K = K // k
    
    if mode == "baseline":
        ddr_reads_A = tiles_M * tiles_K * tiles_N * (m * k * in_bytes)
    else:
        ddr_reads_A = tiles_M * tiles_K * (m * k * in_bytes)
        
    ddr_reads_B = tiles_M * tiles_K * tiles_N * (k * n * in_bytes)
    ddr_writes_C = tiles_M * tiles_N * (m * n * c_bytes)
    
    return ddr_reads_A + ddr_reads_B + ddr_writes_C

def get_hardware_throughput(M, K, N, m, k, n, use_poc, opt_perf, dtype):
    """
    Invokes the underlying MLIR-AIE hardware compilation and execution pipeline.
    Cleans the previous build artifacts to prevent caching errors, then parses GOPS.
    """
    # --- HARDWARE CLEAN RECOVERY BLOCK ---
    # Force removal of old build artifacts to ensure the new matrix shape and data type 
    # are correctly compiled from scratch by the Peano compiler.
    try:
        if os.path.exists("../whole_array.exe"):
            os.remove("../whole_array.exe")
        if os.path.exists("../_build"):
            # Using rm -rf via subprocess to fully purge the build directory safely
            subprocess.run(["rm", "-rf", "../_build"], check=True)
    except Exception as e:
        print(f"      [Warning] Build cleanup failed: {e}")

    print(f"      [Hardware Execution] Target Tile {m}x{k}x{n} (use_poc={use_poc})... ", end="", flush=True)
    
    out_type = "bf16" if dtype == "bf16" else "i32"
    
    cmd = ["make", "run_plot", f"M={M}", f"K={K}", f"N={N}", f"m={m}", f"k={k}", f"n={n}", 
           f"use_poc={use_poc}", "ITERATIONS=15", f"opt_perf={opt_perf}", f"dtype_in={dtype}", f"dtype_out={out_type}"]
    
    try:
        res = subprocess.run(cmd, cwd="..", capture_output=True, text=True, timeout=150)
        gops = 0.0
        for line in res.stdout.split('\n'):
            if "Final Average Throughput:" in line:
                try:
                    gops = float(line.split(":")[1].split()[0])
                except ValueError:
                    gops = 0.0
        return gops
    except Exception:
        return 0.0

def run_embedded_super_tuner(M, K, N, dtype):
    """
    Runs the exhaustive Stochastic Differential Evolution engine for the current shape.
    Finds the absolute optimal alpha, gamma, and sigma weights via hardware-in-the-loop tests.
    """
    print(f"   [Super-Tuner] Starting live Differential Evolution optimization loop...")
    tile_benchmark_cache = {}

    def objective_function(params):
        a, g, s = params
        best_config, _ = solve_mapping(M, K, N, alpha=a, gamma=g, sigma=s, dtype=dtype)
        if not best_config:
            return 0.0
        m, k, n = best_config[0], best_config[1], best_config[2]
        tile_key = f"{m}x{k}x{n}"

        if tile_key in tile_benchmark_cache:
            return -tile_benchmark_cache[tile_key]

        # Interbencmark live execution during optimization search
        gops = get_hardware_throughput(M, K, N, m, k, n, use_poc="1", opt_perf="1", dtype=dtype)
        tile_benchmark_cache[tile_key] = gops
        return -gops 

    bounds = [(80.0, 250.0), (50.0, 200.0), (10.0, 50.0)]
    
    # Differential Evolution tuning parameters matching main_launcher_wBenchmark.py
    result = differential_evolution(objective_function, bounds, strategy='best1bin', 
                                    maxiter=15, popsize=10, tol=0.05)

    best_a, best_g, best_s = result.x
    optimized_gops = float(-result.fun)
    
    # Retrieve the final tile mapping derived from the tuned weights
    best_config, _ = solve_mapping(M, K, N, alpha=best_a, gamma=best_g, sigma=best_s, dtype=dtype)
    
    return best_config[0], best_config[1], best_config[2], optimized_gops

def generate_combinatorial_test_matrix():
    """
    Generates a dense combinatorial grid of matrix shapes (M x K x N).
    Filters shapes dynamically to guarantee physical divisibility constraints.
    """
    dimensions = [64, 128, 256, 512, 1024, 2048, 4096]
    valid_shapes = []
    
    # 1. Balanced and Symmetric Shapes Sweep
    for M in dimensions:
        for K in dimensions:
            for N in dimensions:
                m_c, k_c, n_c, _ = find_candidates(M, K, N, dtype="bf16", cols=8)
                if m_c and k_c and n_c:
                    valid_shapes.append((M, K, N))
                    
    # 2. Asymmetric LLM/Attention Workloads (Small M execution phases)
    llm_M_sizes = [1, 16, 32]
    llm_K_N_sizes = [512, 1024, 2048, 4096]
    for M in llm_M_sizes:
        for K in llm_K_N_sizes:
            for N in llm_K_N_sizes:
                m_c, k_c, n_c, _ = find_candidates(M, K, N, dtype="bf16", cols=8)
                if m_c and k_c and n_c:
                    if (M, K, N) not in valid_shapes:
                        valid_shapes.append((M, K, N))
                        
    return valid_shapes

def main():
    # data_types = ["bf16", "i16"]
    data_types = ["i16"]
    shapes = generate_combinatorial_test_matrix()
    print(f"[PERFECTIONIST TESTER] Initializing massive sweep across {len(shapes)} unique tensor architectures.")

    # Initialize CSV Database Log file with clean academic headers
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "M", "K", "N", "Data_Type",
            "Baseline_m", "Baseline_k", "Baseline_n", "Baseline_GOPS", "Baseline_DDR_Bytes",
            "Opt_m", "Opt_k", "Opt_n", "Opt_GOPS", "Opt_DDR_Bytes",
            "Throughput_Speedup", "DDR_Traffic_Reduction"
        ])

    for dtype in data_types:
        print(f"\n============================================================")
        print(f"STARTING LIVE STOCHASTIC TUNING FOR DATA TYPE: {dtype.upper()}")
        print(f"============================================================")
        
        for M, K, N in shapes:
            print(f"\n[Sweeper Engine] Target Global Tensor Layout: {M}x{K}x{N} ({dtype})")
            
            # Step 1: Silicon-Level Hardware Evaluation - Baseline (Static 32x32x32)
            print(f"   [Baseline Verification] Running static mapping benchmark... ", end="", flush=True)
            gops_baseline = get_hardware_throughput(M, K, N, 32, 32, 32, use_poc="0", opt_perf="0", dtype=dtype)
            print(f"{gops_baseline} GOPS")
            
            # Step 2: Live Embedded Optimization via Differential Evolution Super-Tuner
            opt_m, opt_k, opt_n, gops_optimized = run_embedded_super_tuner(M, K, N, dtype=dtype)
            print(f"   [Optimization Complete] Best Tile Found: {opt_m}x{opt_k}x{opt_n} -> Maximum Performance: {gops_optimized} GOPS")
            
            # Step 3: Extract Offline Memory Traffic Statistics
            bytes_baseline = calculate_analytical_bytes(M, K, N, 32, 32, 32, dtype, "baseline")
            bytes_optimized = calculate_analytical_bytes(M, K, N, opt_m, opt_k, opt_n, dtype, "optimized")
            
            # Step 4: Compute Relative Acceleration and Traffic Mitigation Ratios
            # If the baseline is too small to be profiled by the hardware registers (0.0 GOPS),
            # we log None or 0.0 to prevent skewing the speedup charts with false 1.0x values.
            speedup_gops = gops_optimized / gops_baseline if gops_baseline > 0 else 0.0
            reduction_ddr = bytes_baseline / bytes_optimized if bytes_optimized > 0 else 1.0
            
            # Step 5: Append Records directly into the CSV Database
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    M, K, N, dtype,
                    32, 32, 32, gops_baseline, bytes_baseline,
                    opt_m, opt_k, opt_n, gops_optimized, bytes_optimized,
                    round(speedup_gops, 2), round(reduction_ddr, 2)
                ])
                print(f"   [Data Logger] Row written to database. Speedup: {speedup_gops:.2f}x | DDR Saving: {reduction_ddr:.2f}x")
                
    print(f"\n" + "="*60)
    print(f"[COMPLETED] Comprehensive Database Compiled inside: {CSV_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()