import os
import matplotlib.pyplot as plt
import numpy as np

def generate_unified_memory_plot(M, K, N, dtype, opt_m, opt_k, opt_n):
    """
    Calcola analiticamente gli accessi DDR e genera un grafico a barre raggruppate
    che confronta direttamente la Baseline AMD e l'ottimizzazione Super-Tuner.
    """
    in_bytes = 2 
    c_bytes = 2 if dtype == "bf16" else 4
    to_mb = 1024 * 1024
    
    # --- CALCOLO BASELINE (Fisso a 32x32x32) ---
    b_tiles_M, b_tiles_N, b_tiles_K = M // 32, N // 32, K // 32
    b_reads_A = b_tiles_M * b_tiles_K * b_tiles_N * (32 * 32 * in_bytes)
    b_reads_B = b_tiles_M * b_tiles_K * b_tiles_N * (32 * 32 * in_bytes)
    b_writes_C = b_tiles_M * b_tiles_N * (32 * 32 * c_bytes)
    b_total = b_reads_A + b_reads_B + b_writes_C
    
    # --- CALCOLO OPTIMIZED (Tile dinamico + Stationary A) ---
    o_tiles_M, o_tiles_N, o_tiles_K = M // opt_m, N // opt_n, K // opt_k
    o_reads_A = o_tiles_M * o_tiles_K * (opt_m * opt_k * in_bytes) # Addio loop su N!
    o_reads_B = o_tiles_M * o_tiles_K * o_tiles_N * (opt_k * opt_n * in_bytes)
    o_writes_C = o_tiles_M * o_tiles_N * (opt_m * opt_n * c_bytes)
    o_total = o_reads_A + o_reads_B + o_writes_C
    
    # --- PREPARAZIONE DATI PER IL GRAFICO ---
    categories = ['Matrix A Reads', 'Matrix B Reads', 'Matrix C Writes', 'Total DDR Traffic']
    baseline_vols = [b_reads_A/to_mb, b_reads_B/to_mb, b_writes_C/to_mb, b_total/to_mb]
    opt_vols = [o_reads_A/to_mb, o_reads_B/to_mb, o_writes_C/to_mb, o_total/to_mb]
    
    # --- DISEGNO DEL GRAFICO (Grouped Bar Chart) ---
    output_dir = "memory_profiles"
    os.makedirs(output_dir, exist_ok=True)
    
    x = np.arange(len(categories))
    width = 0.35  # Spessore delle barre
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Barre Rosse (Baseline) e Verdi (Ottimizzato)
    rects1 = ax.bar(x - width/2, baseline_vols, width, label='AMD Baseline (32x32x32)', color='#e63946', edgecolor='black')
    rects2 = ax.bar(x + width/2, opt_vols, width, label=f'Stationary-A ({opt_m}x{opt_k}x{opt_n})', color='#2a9d8f', edgecolor='black')
    
    # Stile e Testi
    ax.set_ylabel('Data Transferred Volume (Megabytes)', fontsize=12, fontweight='bold')
    ax.set_title(f'Analytical Memory Access Profile (DDR) Comparison\nGlobal Shape: {M}x{K}x{N} ({dtype})', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Funzione per aggiungere i numeretti sopra le barre
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f} MB',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    
    # Salvataggio
    output_name = f"{output_dir}/unified_memory_profile_{M}x{K}x{N}_{dtype}.png"
    plt.savefig(output_name)
    plt.close()
    print(f"[Plotter] Images saved in: {output_name}")

# Esempio di utilizzo diretto (generazione multipla per la tesi)
if __name__ == "__main__":
    
    # 1. Caso BERT/LLM (Ottimizzazione eccellente)
    generate_unified_memory_plot(M=512, K=768, N=768, dtype="bf16", opt_m=32, opt_k=16, opt_n=64)
    
    # 2. Caso Matrice Asimmetrica Estrema (Enorme risparmio di banda)
    generate_unified_memory_plot(M=64, K=1024, N=4096, dtype="i16", opt_m=8, opt_k=4, opt_n=512)
    
    # 3. Caso Balanced/Quadrato (Dove AMD si difende bene)
    generate_unified_memory_plot(M=1024, K=1024, N=1024, dtype="bf16", opt_m=32, opt_k=32, opt_n=32)