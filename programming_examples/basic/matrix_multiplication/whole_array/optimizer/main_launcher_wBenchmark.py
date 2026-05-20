import argparse
import subprocess
import json
import os
import sys
import numpy as np
from scipy.optimize import differential_evolution
from tiling_optimizer import solve_mapping

CACHE_FILE = "best_weights_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            try:
                return json.load(f)
            except:
                return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=4)

def run_super_tuner(M, K, N, dtype):
    """Core tuning engine updated to support data types dynamically."""
    print(f"\n" + "="*60)
    print(f"[SUPER-TUNER] Continuous search for shape {M}x{K}x{N} ({dtype})")
    print("="*60)
    
    tile_benchmark_cache = {}
    cache = load_cache()

    def objective_function(params):
        a, g, s = params
        best_config, _ = solve_mapping(M, K, N, alpha=a, gamma=g, sigma=s, dtype=dtype)
        m, k, n = best_config[0], best_config[1], best_config[2]
        tile_key = f"{m}x{k}x{n}"

        if tile_key in tile_benchmark_cache:
            return -tile_benchmark_cache[tile_key]

        print(f"  > Hardware Test: {tile_key} (alpha={a:.2f}, gamma={g:.3f}, sigma={s:.1f})", end="... ", flush=True)
        
        dtype_out = "bf16" if dtype == "bf16" else "i32"

        # Gestione intelligente della cancellazione anche durante il tuning intermedio
        if cache.get("last_compiled_dtype") != dtype:
            try:
                if os.path.exists("../whole_array.exe"): os.remove("../whole_array.exe")
                if os.path.exists("../_build"): subprocess.run(["rm", "-rf", "../_build"], check=True)
                cache["last_compiled_dtype"] = dtype
                save_cache(cache)
            except: pass

        cmd = ["make", "run_plot", f"M={M}", f"K={K}", f"N={N}", f"m={m}", f"k={k}", f"n={n}", 
               "use_poc=1", "ITERATIONS=10", "opt_perf=1", f"dtype_in={dtype}", f"dtype_out={dtype}"]
        res = subprocess.run(cmd, cwd="..", capture_output=True, text=True)
        
        gops = 0.0
        for line in res.stdout.split('\n'):
            if "Final Average Throughput:" in line:
                try:
                    gops = float(line.split(":")[1].split()[0])
                except: gops = 0.0
        
        print(f"{gops} GOPS")
        tile_benchmark_cache[tile_key] = gops
        return -gops 

    bounds = [(80.0, 250.0), (50.0, 200.0), (10.0, 50.0)]
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

def run_test(M, K, N, tag, mode, cache, dtype):
    print(f"\nTEST {tag}: {M}x{K}x{N} ({mode}) - Type: {dtype}")
    
    if mode == "optimized":
        shape_key = f"{M}x{K}x{N}_{dtype}"
        if shape_key not in cache:
            w_best = run_super_tuner(M, K, N, dtype)
            cache = load_cache()  
            cache[shape_key] = w_best
            save_cache(cache)
        
        w = cache[shape_key]
        best_config, _ = solve_mapping(M, K, N, alpha=w['alpha'], gamma=w['gamma'], sigma=w['sigma'], dtype=dtype)
        m, k, n = best_config[0], best_config[1], best_config[2]
        use_poc, opt_perf = "1", "1"
    else:
        m, k, n, use_poc, opt_perf = 32, 32, 32, "0", "0"

    # --- CANCELLAZIONE INTELLIGENTE CONDIZIONALE AVANZATA ---
    last_dtype = cache.get("last_compiled_dtype", "")
    last_mode = cache.get("last_compiled_mode", "")
    
    if last_dtype != dtype or last_mode != mode:
        print(f"[Launcher] Cambio configurazione rilevato (Type: {last_dtype}->{dtype} | Mode: {last_mode}->{mode}). Pulizia di sicurezza...")
        try:
            if os.path.exists("../whole_array.exe"):
                os.remove("../whole_array.exe")
            if os.path.exists("../_build"):
                subprocess.run(["rm", "-rf", "../_build"], check=True)
            
            cache["last_compiled_dtype"] = dtype
            cache["last_compiled_mode"] = mode
            save_cache(cache)
        except Exception as e:
            print(f"[Warning] Errore durante la pulizia condizionale: {e}")
    else:
        print(f"[Launcher] Stessa configurazione hardware precedente ({dtype} - {mode}). Salto la rigenerazione di _build.")

    dtype_out = "bf16" if dtype == "bf16" else "i32"

    # Esecuzione finale dell'hardware
    cmd = ["make", "run_plot", f"M={M}", f"K={K}", f"N={N}", f"m={m}", f"k={k}", f"n={n}", 
           f"use_poc={use_poc}", "ITERATIONS=20", f"opt_perf={opt_perf}", f"dtype_in={dtype}", f"dtype_out={dtype}"]
        
    subprocess.run(cmd, cwd="..", check=True)

# --- BLOCCO MANCANTE DI EXECUTION ENTRY-POINT RIPRISTINATO ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance Launcher for AIE Matrix Multiplication")
    parser.add_argument("-b", "--baseline", action="store_true", help="Run baseline benchmarks")
    parser.add_argument("-o", "--optimized", action="store_true", help="Run optimized benchmarks with Super-Tuner")
    parser.add_argument("-t", "--type", type=str, choices=["i16", "bf16"], default="bf16", 
                        help="Data type inside the matrix: i16 (integers) or bf16 (floating-point)")
    args = parser.parse_args()

    mode = "optimized" if args.optimized else "baseline"
    cache = load_cache()
    
    tests = [
        (512, 512, 512, "BALANCED"),
        (512, 768, 768, "BERT_SHAPE"),
        (512, 512, 2048, "MEMORY_STRESS_N"),
        (512, 2048, 512, "REDUCTION_STRESS_K")
    ]

    for M, K, N, tag in tests:
        run_test(M, K, N, tag, mode, cache, dtype=args.type)