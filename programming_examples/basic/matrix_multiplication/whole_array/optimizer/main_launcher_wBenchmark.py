import argparse
import subprocess
import json
import os
import sys
import numpy as np
from scipy.optimize import differential_evolution
from tiling_optimizer import solve_mapping

# File to store optimized weights for different matrix shapes to avoid redundant tuning
CACHE_FILE = "best_weights_cache.json"

def load_cache():
    """Load the best weights from a JSON file if it exists."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    """Save the updated weights cache to the JSON file."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=4)

def run_super_tuner(M, K, N):
    """
    Core tuning engine using Differential Evolution to find the best heuristic weights.
    It explores the continuous space of (alpha, gamma, sigma) and benchmarks the 
    resulting tiles on actual hardware.
    """
    print(f"\n" + "="*60)
    print(f"[SUPER-TUNER] Continuous search for shape {M}x{K}x{N}")
    print("="*60)
    
    # Internal cache to avoid re-running hardware benchmarks if different weights 
    # point to the same (m, k, n) tiling configuration.
    tile_benchmark_cache = {}

    def objective_function(params):
        """
        Function minimized by the optimizer. 
        It returns negative GOPS because differential_evolution performs minimization.
        """
        a, g, s = params
        # Get the tile suggested by the heuristic with current weight parameters
        best_config, _ = solve_mapping(M, K, N, alpha=a, gamma=g, sigma=s)
        m, k, n = best_config[0], best_config[1], best_config[2]
        tile_key = f"{m}x{k}x{n}"

        # If this tile was already benchmarked, return cached GOPS
        if tile_key in tile_benchmark_cache:
            return -tile_benchmark_cache[tile_key]

        print(f"  > Hardware Test: {tile_key} (alpha={a:.2f}, gamma={g:.3f}, sigma={s:.1f})", end="... ", flush=True)
        
        # Execute hardware benchmark via Makefile. 
        # Uses 10 iterations during tuning to balance speed and accuracy.
        cmd = ["make", "run_plot", f"M={M}", f"K={K}", f"N={N}", f"m={m}", f"k={k}", f"n={n}", 
               "use_poc=1", "ITERATIONS=10", "opt_perf=1"]
        res = subprocess.run(cmd, cwd="..", capture_output=True, text=True)
        
        gops = 0.0
        # Parse stdout to extract the Final Average Throughput
        for line in res.stdout.split('\n'):
            if "Final Average Throughput:" in line:
                try:
                    gops = float(line.split(":")[1].split()[0])
                except: gops = 0.0
        
        print(f"{gops} GOPS")
        tile_benchmark_cache[tile_key] = gops
        return -gops 

    # SEARCH BOUNDS for heuristic weights:
    # Alpha: Arithmetic Intensity (Bandwidth)
    # Gamma: Core efficiency (SIMD/Tiling balance)
    # Sigma: Host Overhead control
    bounds = [
        (80.0, 250.0),   
        (50.0, 200.0),   
        (10.0, 50.0)     
    ]

    # Differential Evolution Algorithm: robust global optimizer for non-linear search spaces.
    result = differential_evolution(objective_function, bounds, strategy='best1bin', 
                                    maxiter=15, popsize=10, tol=0.05)

    best_a, best_g, best_s = result.x
    return {
        "alpha": float(best_a),
        "gamma": float(best_g),
        "sigma": float(best_s),
        "gops": float(-result.fun),
        "tile": "optimized_via_super_tuner"
    }

def run_test(M, K, N, tag, mode, cache):
    """
    Executes the final benchmark for a specific matrix shape.
    Handles both baseline (fixed 32x32x32) and optimized (super-tuner) modes.
    """
    print(f"\nTEST {tag}: {M}x{K}x{N} ({mode})")
    
    if mode == "optimized":
        shape_key = f"{M}x{K}x{N}"
        
        # Check if optimized weights for this shape are already in cache
        if shape_key not in cache:
            w_best = run_super_tuner(M, K, N)
            cache[shape_key] = w_best
            save_cache(cache)
        
        w = cache[shape_key]
        # Retrieve the best tiling using the optimized weights
        best_config, _ = solve_mapping(M, K, N, alpha=w['alpha'], gamma=w['gamma'], sigma=w['sigma'])
        m, k, n = best_config[0], best_config[1], best_config[2]
        use_poc, opt_perf = "1", "1"
    else:
        # Baseline mode: uses fixed AMD-default tiling
        m, k, n, use_poc, opt_perf = 32, 32, 32, "0", "0"

    # Final Hardware Execution:
    # Averaged over 20 iterations to ensure stable and statistically relevant results[cite: 19].
    cmd = ["make", "run_plot", f"M={M}", f"K={K}", f"N={N}", f"m={m}", f"k={k}", f"n={n}", 
               f"use_poc={use_poc}", "ITERATIONS=20", f"opt_perf={opt_perf}"] 
    
    subprocess.run(cmd, cwd="..", check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance Launcher for AIE Matrix Multiplication")
    parser.add_argument("-b", "--baseline", action="store_true", help="Run baseline benchmarks")
    parser.add_argument("-o", "--optimized", action="store_true", help="Run optimized benchmarks with Super-Tuner")
    args = parser.parse_args()

    mode = "optimized" if args.optimized else "baseline"
    cache = load_cache()
    
    # Test Suite: covering balanced, real-world (BERT), and edge-case (Stress) scenarios
    tests = [
        (512, 512, 512, "BALANCED"),
        (512, 768, 768, "BERT_SHAPE"),
        (512, 512, 2048, "MEMORY_STRESS_N"),
        (512, 2048, 512, "REDUCTION_STRESS_K")
    ]

    for M, K, N, tag in tests:
        run_test(M, K, N, tag, mode, cache)