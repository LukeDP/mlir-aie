import sys
import subprocess
import os
from tiling_optimizer import solve_mapping

def run_test(M, K, N, tag):
    print(f"\n{'='*60}")
    print(f"TEST {tag}: Dimensions {M}x{K}x{N}")
    print(f"{'='*60}")

    # 1. Optimization
    best_config, _ = solve_mapping(M, K, N)
    if not best_config:
        print("No valid Configurations")
        return
    m, k, n, j, ai = best_config
    print(f"-> Optimal Mapping: m={m}, k={k}, n={n} | AI: {ai:.2f}")

    # 2. Makefile
    cmd = [
        "make", "run_plot",
        f"M={M}", f"K={K}", f"N={N}",
        f"m={m}", f"k={k}", f"n={n}",
        "n_aie_cols=8", "use_poc=1", "ITERATIONS=5"
    ]

    try:
        
        process = subprocess.run(cmd, cwd="..", capture_output=True, text=True, check=True)
        # Results extraction
        for line in process.stdout.split('\n'):
            if "Avg NPU gflops" in line or "PASS!" in line:
                print(f"   {line}")
    except subprocess.CalledProcessError as e:
        print(f"Error: \n{e.stderr}")

if __name__ == "__main__":

    tests = [
        (512, 512, 512, "BALANCED"),
        (512, 768, 768, "BERT_SHAPE"),
        (512, 512, 2048, "MEMORY_STRESS_N"),
        (512, 2048, 512, "REDUCTION_STRESS_K")
    ]
    for M, K, N, tag in tests:
        run_test(M, K, N, tag)