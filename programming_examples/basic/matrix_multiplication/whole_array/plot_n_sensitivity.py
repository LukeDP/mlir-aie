import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(filename, label_to_find):
    if not os.path.exists(filename): 
        print(f"File {filename} non trovato.")
        return [], []
    df = pd.read_csv(filename, names=['M', 'K', 'N', 'AI', 'GOPS', 'cols', 'Label'])
    # Filtriamo per M=512, K=512, 8 colonne e la Label specifica
    mask = (df['Label'] == label_to_find) & (df['cols'] == 8) & (df['M'] == 512) & (df['K'] == 512)
    filtered = df[mask].sort_values(by='N')
    if filtered.empty: return [], []
    final = filtered.groupby('N')['GOPS'].mean().sort_index()
    return final.index.tolist(), final.values.tolist()

def calc_traffic(M, K, N, cols, n_tile, label):
    dsize = 2 # int16 -> 2 bytes
    if label == "Baseline":
        # Matrix A is re-fetched N / (n_tile * cols) times
        redundancy = max(1, N / (n_tile * cols))
        return ((M * K * redundancy) + (K * N) + (M * N)) * dsize / (1024**2)
    else: # PoC (Stationary A)
        # Matrix A is fetched only once
        return ((M * K) + (K * N) + (M * N)) * dsize / (1024**2)

# Caricamento dati
N_vals, baseline_gops = load_data('performance_log.csv', 'Baseline')
N_poc, poc_gops = load_data('performance_log.csv', 'PoC')

if not N_vals and not N_poc:
    print("Nessun dato trovato nel log. Verifica di aver eseguito i test.")
    exit()

fig, ax1 = plt.subplots(figsize=(12, 7))

# Configurazione Asse X (N)
ax1.set_xlabel('N Dimension (Matrix B columns)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Throughput (GOPS)', color='blue', fontsize=12, fontweight='bold')

# Plot dei GOPS
lns1 = []
if N_vals:
    l1 = ax1.plot(N_vals, baseline_gops, 'o-', color='cornflowerblue', label='Baseline GOPS', linewidth=2, markersize=8)
    lns1 += l1
if N_poc:
    l2 = ax1.plot(N_poc, poc_gops, 's-', color='blue', label='PoC GOPS', linewidth=3, markersize=10)
    lns1 += l2

ax1.set_xscale('log', base=2)
all_n = sorted(list(set(N_vals + N_poc)))
ax1.set_xticks(all_n)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.grid(True, which="both", ls="-", alpha=0.2)
ax1.tick_params(axis='y', labelcolor='blue')

# Configurazione Asse Y2 (Traffico DDR)
ax2 = ax1.twinx()
ax2.set_ylabel('DDR Traffic (MB)', color='red', fontsize=12, fontweight='bold')

lns2 = []
if N_vals:
    bt = [calc_traffic(512, 512, n, 8, 32, "Baseline") for n in N_vals]
    l3 = ax2.plot(N_vals, bt, 'o--', color='lightcoral', label='Baseline Traffic (MB)', alpha=0.8)
    lns2 += l3
if N_poc:
    pt = [calc_traffic(512, 512, n, 8, 32, "PoC") for n in N_poc]
    l4 = ax2.plot(N_poc, pt, 'v--', color='red', label='PoC Traffic (MB)', linewidth=2)
    lns2 += l4

ax2.tick_params(axis='y', labelcolor='red')

# --- UNIONE DELLE LEGENDE ---
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left', frameon=True, shadow=True, fontsize=10)

plt.title('N-Sensitivity Analysis: Throughput vs. DDR Traffic\n(Fixed M=512, K=512, 8 columns)', fontsize=14, fontweight='bold')

# Salvataggio con margini ottimizzati
plt.tight_layout()
plt.savefig('n_sensitivity_analysis.png', bbox_inches='tight', dpi=300)
print("Grafico generato con successo: n_sensitivity_analysis.png")