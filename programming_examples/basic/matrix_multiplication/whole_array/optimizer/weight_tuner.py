import subprocess
import pandas as pd
import sys
from tiling_optimizer import solve_mapping

# 1. Definizione dello spazio di ricerca[cite: 28]
alphas = [10.0, 50.0, 100.0]   # Spingiamo sull'efficienza
gammas = [0.01, 0.1, 0.5]      # Teniamo bassa la penalità contenzione
sigmas = [50.0, 100.0, 200.0, 500.0] # Testiamo diverse tolleranze all'overhead



# Dimensioni del test case per il tuning
#(BALANCED)
# M, K, N = 512, 512, 512
#(BERT_SHAPE)
M, K, N = 512, 768, 768


results = []
tile_cache = {} # Evita di rieseguire test hardware per tile già visti

print(f"Inizio tuning su matrice {M}x{K}x{N}...")

for a in alphas:
    for g in gammas:
        for s in sigmas:
            # Otteniamo il tile suggerito dall'euristica con i pesi correnti[cite: 28]
            best_config, _ = solve_mapping(M, K, N, alpha=a, gamma=g, sigma=s)
            m, k, n = best_config[0], best_config[1], best_config[2]
            
            tile_key = (m, k, n)
            
            # Se questo tile non è mai stato testato, eseguiamo il benchmark[cite: 27]
            if tile_key not in tile_cache:
                print(f"\n--- Testing nuovo tile: {m}x{k}x{n} (alpha={a}, gamma={g}, sigma={s}) ---")
                
                # Chiamata al Makefile nella cartella superiore[cite: 27]
                cmd = [
                    "make", "run_plot", 
                    f"M={M}", f"K={K}", f"N={N}", 
                    f"m={m}", f"k={k}", f"n={n}", 
                    "use_poc=1", "ITERATIONS=3" # 3 iterazioni bastano per il tuning
                ]
                
                try:
                    res = subprocess.run(cmd, cwd="..", capture_output=True, text=True, check=True)
                    
                    # Estrazione dei GOPS dall'output[cite: 27]
                    gops = 0.0
                    for line in res.stdout.split('\n'):
                        if "Final Average Throughput:" in line:
                            gops = float(line.split(":")[1].split()[0])
                    
                    tile_cache[tile_key] = gops
                except subprocess.CalledProcessError:
                    print(f"Errore durante l'esecuzione del tile {m}x{k}x{n}. Salto...")
                    tile_cache[tile_key] = 0.0
            
            # Registriamo il risultato per questa combinazione di pesi[cite: 28]
            results.append({
                "alpha": a, 
                "gamma": g, 
                "sigma": s, 
                "tile": f"{m}x{k}x{n}", 
                "GOPS": tile_cache[tile_key]
            })

# 2. Salvataggio ed esportazione dei risultati[cite: 27]
df = pd.DataFrame(results)
df.to_csv("weight_tuning_results.csv", index=False)

print("\n" + "="*30)
print("TUNING COMPLETATO")
print("Risultati salvati in: weight_tuning_results.csv")
print("="*30)

# Mostra le migliori 5 combinazioni
print("\nTop 5 configurazioni trovate:")
print(df.sort_values(by="GOPS", ascending=False).head(5))