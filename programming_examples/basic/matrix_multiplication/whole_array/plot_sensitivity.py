import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. FUNZIONE DI CARICAMENTO AUTOMATICO ---
def load_data_from_log(filename, label_to_find):
    """Legge i dati dal CSV filtrando per etichetta e configurazione."""
    if not os.path.exists(filename):
        print(f"Attenzione: file {filename} non trovato.")
        return [], []
    
    try:
        # Leggiamo il log. Le vecchie righe con 6 colonne avranno Label=NaN
        df = pd.read_csv(filename, names=['M', 'K', 'N', 'AI', 'GOPS', 'cols', 'Label'])
        
        # Filtriamo per design (Baseline/PoC), 8 colonne e dimensione M=512 fissa
        mask = (df['Label'] == label_to_find) & (df['cols'] == 8) & (df['M'] == 512)
        filtered = df[mask].sort_values(by='K')
        
        if filtered.empty:
            print(f"Nessun dato trovato per '{label_to_find}' (M=512, cols=8).")
            return [], []
            
        # Raggruppiamo per K per gestire eventuali test ripetuti (prendendo la media)
        final = filtered.groupby('K')['GOPS'].mean().sort_index()
        return final.index.tolist(), final.values.tolist()
    except Exception as e:
        print(f"Errore durante la lettura del log: {e}")
        return [], []

# --- 2. CARICAMENTO DATI ---
K_vals, baseline_gops = load_data_from_log('performance_log_forPlotSensitivity.csv', 'Baseline')
K_poc, poc_gops = load_data_from_log('performance_log_forPlotSensitivity.csv', 'PoC')

# Controllo integrità dati
if not K_vals and not K_poc:
    print("Nessun dato caricato. Assicurati di aver rigenerato il log con il nuovo Makefile.")
    exit()

# --- 3. PARAMETRI ARCHITETTONICI ---
M, N = 512, 512
n, cols = 32, 8
dsize_in = 2  # i16 (2 bytes)
dsize_out = 4 # i32 (4 bytes)

# Data Redundancy Factor per la Baseline (Ra = N / (n * cols) = 2)
Ra = N / (n * cols)

# --- 4. CALCOLO DEL TRAFFICO DDR (MB) ---
def calc_traffic_mb(K, is_poc):
    # Nella Baseline, la Matrice A viene ri-letta dalla DDR con fattore Ra
    # Nella PoC (Stationary-A), la Matrice A viene letta una sola volta
    red_factor = 1 if is_poc else Ra
    bytes_a = M * K * red_factor * dsize_in
    bytes_b = K * N * dsize_in
    bytes_c = M * N * dsize_out
    total_bytes = bytes_a + bytes_b + bytes_c
    return total_bytes / (1024**2) # Conversione in MiB

baseline_traffic = [calc_traffic_mb(k, False) for k in K_vals]
poc_traffic = [calc_traffic_mb(k, True) for k in K_poc]

# --- 5. GENERAZIONE DEL GRAFICO COMBINATO ---
fig, ax1 = plt.subplots(figsize=(12, 7))

# Asse primario (Sinistra): Throughput in GOPS
ax1.set_xlabel('K Dimension', fontsize=12, fontweight='bold')
ax1.set_ylabel('Performance (GOPS)', color='blue', fontsize=12, fontweight='bold')

if K_vals:
    ax1.plot(K_vals, baseline_gops, 'o-', color='cornflowerblue', label='Baseline Throughput', linewidth=2, markersize=8, alpha=0.7)
if K_poc:
    ax1.plot(K_poc, poc_gops, 's-', color='blue', label='Optimized PoC Throughput', linewidth=3, markersize=10)

ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xscale('log', base=2)
# Uniamo i valori di K trovati per le tacche dell'asse X
all_k = sorted(list(set(K_vals + K_poc)))
ax1.set_xticks(all_k)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.grid(True, which="both", ls="-", alpha=0.2)

# Asse secondario (Destra): Traffico DDR in MB
ax2 = ax1.twinx()
ax2.set_ylabel('DDR Traffic (MB)', color='red', fontsize=12, fontweight='bold')

if K_vals:
    ax2.plot(K_vals, baseline_traffic, 'o--', color='lightcoral', label='Baseline DDR Traffic', alpha=0.7)
if K_poc:
    ax2.plot(K_poc, poc_traffic, 'v--', color='red', label='PoC DDR Traffic', linewidth=2, markersize=8)

ax2.tick_params(axis='y', labelcolor='red')

# Legenda combinata
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, shadow=True)

plt.title('Performance vs. DDR Data Volume', fontsize=15, fontweight='bold', pad=20)
fig.tight_layout()

# Salvataggio
plt.savefig('k_sensitivity_combined.png', dpi=300)
print(f"Graph generated: k_sensitivity_combined.png with {len(K_vals)} points Baseline and {len(K_poc)} points PoC.")