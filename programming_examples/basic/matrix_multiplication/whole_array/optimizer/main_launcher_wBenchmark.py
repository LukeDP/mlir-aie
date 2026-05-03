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
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=4)

def run_super_tuner(M, K, N):
    print(f"\n" + "="*60)
    print(f"[SUPER-TUNER] Ricerca continua per {M}x{K}x{N}")
    print("="*60)
    
    # Cache per evitare di rieseguire test hardware se pesi diversi portano allo stesso tile
    tile_benchmark_cache = {}

    def objective_function(params):
        a, g, s = params
        # Chiediamo il tile suggerito con i pesi continui correnti
        best_config, _ = solve_mapping(M, K, N, alpha=a, gamma=g, sigma=s)
        m, k, n = best_config[0], best_config[1], best_config[2]
        tile_key = f"{m}x{k}x{n}"

        if tile_key in tile_benchmark_cache:
            return -tile_benchmark_cache[tile_key]

        print(f"  > Test Hardware: {tile_key} (alpha={a:.2f}, gamma={g:.3f}, sigma={s:.1f})", end="... ", flush=True)
        
        # Esecuzione del benchmark hardware tramite Makefile
        cmd = ["make", "run_plot", f"M={M}", f"K={K}", f"N={N}", f"m={m}", f"k={k}", f"n={n}", 
               "use_poc=1", "ITERATIONS=10", "opt_perf=1"]
        res = subprocess.run(cmd, cwd="..", capture_output=True, text=True)
        
        gops = 0.0
        for line in res.stdout.split('\n'):
            if "Final Average Throughput:" in line:
                try:
                    gops = float(line.split(":")[1].split()[0])
                except: gops = 0.0
        
        print(f"{gops} GOPS")
        tile_benchmark_cache[tile_key] = gops
        return -gops # Minimizziamo il negativo per massimizzare i GOPS

    # Range continui come da tua richiesta
    bounds = [
        (10.0, 100.0),   # Alpha
        (0.01, 0.5),     # Gamma
        (50.0, 500.0)    # Sigma
    ]

    # Ottimizzazione globale (Evoluzione Differenziale)
    # popsize e maxiter controllano quanto è "spinta" la ricerca
    result = differential_evolution(objective_function, bounds, strategy='best1bin', 
                                    maxiter=8, popsize=5, tol=0.05)

    best_a, best_g, best_s = result.x
    return {
        "alpha": float(best_a),
        "gamma": float(best_g),
        "sigma": float(best_s),
        "gops": float(-result.fun),
        "tile": "optimized_via_super_tuner"
    }

def run_test(M, K, N, tag, mode, cache):
    print(f"\nTEST {tag}: {M}x{K}x{N} ({mode})")
    
    if mode == "optimized":
        shape_key = f"{M}x{K}x{N}"
        
        # Logica Una-Tantum con Memoizzazione
        if shape_key not in cache:
            w_best = run_super_tuner(M, K, N)
            cache[shape_key] = w_best
            save_cache(cache)
        
        w = cache[shape_key]
        best_config, _ = solve_mapping(M, K, N, alpha=w['alpha'], gamma=w['gamma'], sigma=w['sigma'])
        m, k, n = best_config[0], best_config[1], best_config[2]
        use_poc = "1"
    else:
        m, k, n, use_poc = 64, 64, 32, "0"

    # Esecuzione finale mediata su 5 iterazioni per il risultato definitivo[cite: 28]
    cmd = ["make", "run_plot", f"M={M}", f"K={K}", f"N={N}", f"m={m}", f"k={k}", f"n={n}", 
               "use_poc=1", "ITERATIONS=20", "opt_perf=1"] 
    subprocess.run(cmd, cwd="..", check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--baseline", action="store_true")
    parser.add_argument("-o", "--optimized", action="store_true")
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
        run_test(M, K, N, tag, mode, cache)