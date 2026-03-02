import matplotlib.pyplot as plt
import numpy as np
import sys, os, csv

# Ordine argomenti dal Makefile: M, K, N, m, k, n, cols, dsize_in, gops
M, K, N, m, k, n, cols, dsize_in, gops = map(float, sys.argv[1:10])
dsize_out = 4 
design_label = sys.argv[10]

# 1. Calcolo Arithmetic Intensity (AI)
ops = 2 * M * K * N
# Il Data Redundancy Factor è N / (n * cols)
bytes_a = M * K * (N / (n * cols)) * dsize_in
bytes_b = K * N * dsize_in
bytes_c = M * N * dsize_out
total_bytes = bytes_a + bytes_b + bytes_c
ai = ops / total_bytes

# 2. Log dei dati
log_file = 'performance_log.csv'
if not os.path.exists(log_file):
    open(log_file, 'w').close()
with open(log_file, 'a') as f:
    writer = csv.writer(f)
    writer.writerow([M, K, N, ai, gops, cols, design_label])

# 3. Plotting
plt.figure(figsize=(12, 8))
peak_perf = 2500 
theo_bw = 35    
ai_range = np.logspace(-1, 3, 1000)

# Tetto Teorico
plt.loglog(ai_range, np.minimum(peak_perf, theo_bw * ai_range), 'r--', alpha=0.3, label='Theo. Peak BW (35 GB/s)')

if os.path.getsize(log_file) > 0:
    data = np.genfromtxt(log_file, delimiter=',')
    if data.ndim == 1: data = data.reshape(1, -1)
    
    # Calcolo Bandwidth Effettiva (Limite Reale)
    max_bw_measured = np.max(data[:, 4] / data[:, 3])
    plt.loglog(ai_range, np.minimum(peak_perf, max_bw_measured * ai_range), 'g:', linewidth=2, label=f'Effective BW ({max_bw_measured:.2f} GB/s)')

    unique_cols = np.unique(data[:, 5])
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(unique_cols)))

    for i, c_val in enumerate(unique_cols):
        subset = data[data[:, 5] == c_val]
        subset = subset[subset[:, 3].argsort()]
        # Disegniamo la linea per la configurazione di colonne
        plt.plot(subset[:, 3], subset[:, 4], 'o-', color=colors[i], label=f'{int(c_val)} AIE Columns', alpha=0.6)
        
        for row in subset:
            # Controllo se è il tile 16x16x16 (AI circa 73.14)
            if row[3] > 118.0:
                # Cambiamo colore al pallino: lo sovrapponiamo rosso
                plt.scatter(row[3], row[4], color='red', s=100, edgecolors='black', zorder=5)
                plt.annotate(f"N={int(row[2])}", (row[3], row[4]), 
                             textcoords="offset points", xytext=(0,15), ha='center', 
                             fontsize=10, color='red', weight='bold')
            else:
                # Etichetta standard. Se K è grande (paradox), mostriamo anche K
                label = f"N={int(row[2])}, K={int(row[1])}" if row[1] > 512 else f"N={int(row[2])}"
                plt.annotate(label, (row[3], row[4]), 
                             textcoords="offset points", xytext=(0,8), ha='center', fontsize=9)

plt.xlabel('Arithmetic Intensity (Ops/Byte)', fontsize=12)
plt.ylabel('Performance (GOPS)', fontsize=12)
plt.title(f'Roofline Analysis: Krackan NPU (i16 Integer Math)', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(loc='lower right')
plt.savefig('roofline_current.png', dpi=300)