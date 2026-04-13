import matplotlib.pyplot as plt
import numpy as np
import sys, os, csv

# 1. Caricamento argomenti dal Makefile
# Ordine: M, K, N, m, k, n, cols, dsize_in, gops, label, out_file
try:
    M, K, N, m, k, n, cols, dsize_in, gops = map(float, sys.argv[1:10])
    design_label = sys.argv[10]
    out_file = sys.argv[11] if len(sys.argv) > 11 else 'roofline_current.png'
except IndexError:
    print("Errore: Argomenti insufficienti. Uso: python3 plot_roofline.py M K N m k n cols dsize_in gops label [out_file]")
    sys.exit(1)

# Parametri fissi per l'architettura Krackan
dsize_out = 4  # i32 accumulation
peak_perf = 2500 # GOPS teorici picco vettoriale
theo_bw = 35    # GB/s teorici banda DDR

# 2. Calcolo Arithmetic Intensity (AI)
ops = 2 * M * K * N
# Il fattore di ridondanza N/(n*cols) rappresenta quante volte Matrix A viene riletta
bytes_a = M * K * (N / (n * cols)) * dsize_in
bytes_b = K * N * dsize_in
bytes_c = M * N * dsize_out
total_bytes = bytes_a + bytes_b + bytes_c
ai = ops / total_bytes

# 3. Logging dei dati nel CSV (Persistente per confronti)
log_file = 'performance_log_heuristics.csv'
file_exists = os.path.isfile(log_file)
with open(log_file, 'a', newline='') as f:
    writer = csv.writer(f)
    # Formato: M, K, N, AI, GOPS, Cols, Label
    writer.writerow([int(M), int(K), int(N), ai, gops, int(cols), design_label])

# 4. Lettura e Filtraggio per il Grafico Corrente
data_to_plot = []
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[6] == design_label: # Filtriamo solo per il design attuale
                data_to_plot.append([float(x) for i, x in enumerate(row) if i < 6])

data_to_plot = np.array(data_to_plot)

# 5. Creazione del Grafico Roofline
plt.figure(figsize=(10, 7))
ai_range = np.logspace(-1, 3, 1000)

# Disegno tetti teorici
plt.loglog(ai_range, np.minimum(peak_perf, theo_bw * ai_range), 'r--', alpha=0.4, label=f'Theo. Peak BW ({theo_bw} GB/s)')

if data_to_plot.size > 0:
    # Calcolo Bandwidth Effettiva misurata (Pendenza della retta dei dati)
    max_bw_measured = np.max(data_to_plot[:, 4] / data_to_plot[:, 3])
    plt.loglog(ai_range, np.minimum(peak_perf, max_bw_measured * ai_range), 'g:', linewidth=2, label=f'Effective BW ({max_bw_measured:.2f} GB/s)')

    # Plot dei punti sperimentali
    unique_cols = np.unique(data_to_plot[:, 5])
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(unique_cols)))

    for i, c_val in enumerate(unique_cols):
        subset = data_to_plot[data_to_plot[:, 5] == c_val]
        subset = subset[subset[:, 3].argsort()] # Ordiniamo per AI per la linea
        plt.plot(subset[:, 3], subset[:, 4], 'o-', color=colors[i], markersize=8, label=f'{int(c_val)} AIE Columns ({design_label})', alpha=0.7)
        
        for row in subset:
            # Etichette dinamiche per i punti
            label = f"N={int(row[2])}"
            if row[1] > 512: label += f",K={int(row[1])}" # Mostriamo K se è lo scenario di stress
            plt.annotate(label, (row[3], row[4]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

# Formattazione finale
plt.xlabel('Arithmetic Intensity (Ops/Byte)', fontsize=12, fontweight='bold')
plt.ylabel('Performance (GOPS)', fontsize=12, fontweight='bold')
plt.title(f'Roofline Analysis: Krackan NPU - {design_label} Mode', fontsize=14, fontweight='bold')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend(loc='lower right', frameon=True)

# Salvataggio
plt.savefig(out_file, dpi=300, bbox_inches='tight')
print(f"Grafico '{design_label}' salvato con successo in: {out_file} (AI: {ai:.2f})")

'''
import matplotlib.pyplot as plt
import numpy as np
import os, csv

def generate_roofline(dataset, output_name, title_suffix=""):
    # 1. Preparazione Dati
    # M, K, N, AI, GOPS, COLS
    data = np.array(dataset)
    
    # 2. Configurazione Estetica per LaTeX (Font più grandi)
    plt.figure(figsize=(8, 6)) # Più compatto
    plt.rcParams.update({'font.size': 14})
    
    peak_perf = 2500 
    theo_bw = 35    
    # Range AI zoomato: da 10 a 300 invece di 0.1 a 1000
    ai_range = np.logspace(0.9, 2.5, 1000) 

    # Tetto Teorico
    plt.loglog(ai_range, np.minimum(peak_perf, theo_bw * ai_range), 'r--', alpha=0.3, label='Theo. Peak BW (35 GB/s)')

    # Calcolo Bandwidth Effettiva (Limite Reale)
    # Usiamo il massimo GOPS/AI del dataset attuale
    max_bw_measured = np.max(data[:, 4] / data[:, 3])
    plt.loglog(ai_range, np.minimum(peak_perf, max_bw_measured * ai_range), 'g:', linewidth=3, label=f'Effective BW ({max_bw_measured:.2f} GB/s)')

    unique_cols = np.unique(data[:, 5])
    colors = ['#440154', '#fde725'] # Colori ad alto contrasto

    for i, c_val in enumerate(unique_cols):
        subset = data[data[:, 5] == c_val]
        subset = subset[subset[:, 3].argsort()]
        plt.plot(subset[:, 3], subset[:, 4], 'o-', color=colors[i], markersize=8, label=f'{int(c_val)} AIE Columns', alpha=0.8)
        
        for row in subset:
            # Highlight per punti critici (BERT o Tiling subottimale)
            # AI > 110 (BERT) o AI circa 73 (Tiling 16x16x16)
            if row[3] > 110.0:
                plt.scatter(row[3], row[4], color='red', s=150, edgecolors='black', zorder=5)
                plt.annotate(f"N={int(row[2])}", (row[3], row[4]), 
                             textcoords="offset points", xytext=(0,12), ha='center', 
                             fontsize=12, color='red', weight='bold')
            else:
                label = f"N={int(row[2])}"
                plt.annotate(label, (row[3], row[4]), 
                             textcoords="offset points", xytext=(0,8), ha='center', fontsize=11)

    # 3. Zoom e Etichette
    plt.xlabel('Arithmetic Intensity (Ops/Byte)', fontsize=15, weight='bold')
    plt.ylabel('Performance (GOPS)', fontsize=15, weight='bold')
    plt.title(f'Roofline Analysis (Zoomed){title_suffix}', fontsize=16, weight='bold')
    
    # Impostiamo limiti stretti per lo ZOOM
    plt.xlim(10, 250)
    plt.ylim(20, 600)
    
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(loc='lower right', fontsize=11, frameon=True)
    
    plt.tight_layout()
    plt.savefig(output_name, dpi=300)
    plt.close()

# --- DATASETS ---
# Dataset 1: Baseline
initial_data = [
    [512,512,128,26.947,54.2952,1.0],
    [512,512,256,26.947,98.9806,1.0],
    [512,512,512,26.947,148.8,1.0],
    [512,2048,1024,29.257,194.29,1.0],
    [512,512,256,102.4,49.2,8.0],
    [512,512,512,102.4,100.237,8.0],
    [512,512,1024,102.4,184.428,8.0],
    [512,512,512,73.142,87.6953,8.0],
    [512,512,2048,102.4,336.702,8.0],
    [512,512,64,26.947,27.1037,1.0]
]

# Dataset 2: Final (Incluso BERT PoC)
final_data = initial_data + [[512,768,768,118.153,179.542,8.0]]

# Generazione immagini
generate_roofline(initial_data, 'roofline_initial_zoomed.png', " - Baseline")
generate_roofline(final_data, 'roofline_final_zoomed.png', " - PoC BERT")

print("Immagini zoomate generate con successo.")
'''