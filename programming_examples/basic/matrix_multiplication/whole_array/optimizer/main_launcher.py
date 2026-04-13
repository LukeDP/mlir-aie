import argparse
import subprocess
from tiling_optimizer import solve_mapping

def run_test(M, K, N, tag, mode):
    print(f"\nTEST {tag}: {M}x{K}x{N} ({mode})")
    
    if mode == "optimized":
        best_config, _ = solve_mapping(M, K, N)
        m, k, n, j, ai = best_config
        use_poc = "1"
        out_png = "roofline_optimized.png"
    else:
        # Standard values for the baseline (m=64, k=64, n=32)
        m, k, n = 64, 64, 32 
        use_poc = "0"
        out_png = "roofline_baseline.png"

    cmd = [
        "make", "run_plot",
        f"M={M}", f"K={K}", f"N={N}",
        f"m={m}", f"k={k}", f"n={n}",
        f"use_poc={use_poc}", f"OUT_PNG={out_png}",
        "ITERATIONS=5"
    ]
    
    subprocess.run(cmd, cwd="..", check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--baseline", action="store_true", help="Run baseline tests")
    parser.add_argument("-o", "--optimized", action="store_true", help="Run optimized PoC tests")
    args = parser.parse_args()

    mode = "optimized" if args.optimized else "baseline"
    
    tests = [
        (512, 512, 512, "BALANCED"),
        (512, 768, 768, "BERT_SHAPE"),
        (512, 512, 2048, "MEMORY_STRESS_N"),
        (512, 2048, 512, "REDUCTION_STRESS_K")
    ]

    for M, K, N, tag in tests:
        run_test(M, K, N, tag, mode)