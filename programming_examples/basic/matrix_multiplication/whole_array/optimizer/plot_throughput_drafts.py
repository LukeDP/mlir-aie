import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

output_dir = "draft_plots"
os.makedirs(output_dir, exist_ok=True)

try:
    df_bf16 = pd.read_csv("dataset_bf16_final.csv")
    df_i16 = pd.read_csv("dataset_i16_final.csv")
    df = pd.concat([df_bf16, df_i16], ignore_index=True)
except FileNotFoundError:
    print("Errore: Assicurati che i file CSV siano nella cartella.")
    exit()

# ==============================================================================
# GRAFICO 1
# ==============================================================================
def plot_scaling_N():
    subset = df[(df['M'] == 128) & (df['K'] == 1024) & (df['Data_Type'] == 'i16')].sort_values(by='N')
    subset = subset[(subset['Baseline_GOPS'] > 0) | (subset['Opt_GOPS'] > 0)]
    
    if subset.empty: return
        
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    x_labels = [str(n) for n in subset['N']]
    x = np.arange(len(x_labels))
    
    ax.plot(x, subset['Baseline_GOPS'], marker='o', color='#e63946', linewidth=2, label='AMD Baseline (32x32x32)')
    ax.plot(x, subset['Opt_GOPS'], marker='s', color='#2a9d8f', linewidth=2, label='Super-Tuner ')
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Matrix Dimension N', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hardware Throughput (GOPS)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Scaling over dimension N (Fixed M=128, K=1024, i16)', fontsize=14, fontweight='bold', pad=15)
    ax.fill_between(x, subset['Baseline_GOPS'], subset['Opt_GOPS'], where=(subset['Opt_GOPS'] >= subset['Baseline_GOPS']), interpolate=True, color='#2a9d8f', alpha=0.15)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_draft_scaling_N.png")
    plt.close()
    
# ==============================================================================
# GRAFICO: Scaling di K (M=128, N=1024)
# ==============================================================================
def plot_scaling_K():
    scale_df = df[(df['M'] == 128) & (df['N'] == 1024) & (df['Data_Type'] == 'i16')].sort_values(by='K')
    scale_df = scale_df[(scale_df['Baseline_GOPS'] > 0) | (scale_df['Opt_GOPS'] > 0)]

    if scale_df.empty: return

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    x_labels = [str(k) for k in scale_df['K']]
    x = np.arange(len(x_labels))

    ax.plot(x, scale_df['Baseline_GOPS'], marker='o', color='#e63946', linewidth=2, label='AMD Baseline (32x32x32)')
    ax.plot(x, scale_df['Opt_GOPS'], marker='s', color='#2a9d8f', linewidth=2, label='Super-Tuner')
    
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Matrix Dimension K', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hardware Throughput (GOPS)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Scaling over dimension K (Fixed M=128, K=1024, i16)', fontsize=14, fontweight='bold', pad=15)
    ax.fill_between(x, scale_df['Baseline_GOPS'], scale_df['Opt_GOPS'], color='#2a9d8f', alpha=0.1)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_draft_scaling_K.png")
    plt.close()

# ==============================================================================
# GRAFICO: Scaling di M (K=128, N=1024)
# ==============================================================================
def plot_scaling_M():
    scale_df = df[(df['K'] == 128) & (df['N'] == 1024) & (df['Data_Type'] == 'i16')].sort_values(by='M')
    scale_df = scale_df[(scale_df['Baseline_GOPS'] > 0) | (scale_df['Opt_GOPS'] > 0)]

    if scale_df.empty: return

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    x_labels = [str(m) for m in scale_df['M']]
    x = np.arange(len(x_labels))
    
    ax.plot(x, scale_df['Baseline_GOPS'], marker='o', color='#e63946', linewidth=2, label='AMD Baseline (32x32x32)')
    ax.plot(x, scale_df['Opt_GOPS'], marker='s', color='#2a9d8f', linewidth=2, label='Super-Tuner')
    
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Matrix Dimension M', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hardware Throughput (GOPS)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Scaling over dimension M (Fixed M=128, K=1024, i16)', fontsize=14, fontweight='bold', pad=15)
    ax.fill_between(x, scale_df['Baseline_GOPS'], scale_df['Opt_GOPS'], color='#2a9d8f', alpha=0.1)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_draft_scaling_M.png")
    plt.close()

# ==============================================================================
# GRAFICO 2: La Rivincita della Baseline (Asse Y Spezzato CONTINUO, senza buchi)
# ==============================================================================
def plot_square_matrices():
    square_df = df[(df['M'] == df['K']) & (df['K'] == df['N']) & (df['Data_Type'] == 'i16')].sort_values(by='M')
    square_df = square_df[(square_df['Baseline_GOPS'] > 0) | (square_df['Opt_GOPS'] > 0)]
    
    if square_df.empty: return

    # Creiamo i due grafici
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6), dpi=150, gridspec_kw={'height_ratios': [1, 1]})
    
    fig.subplots_adjust(hspace=0.0) 
    
    x_labels = [str(m) for m in square_df['M']]
    x = np.arange(len(x_labels))
    width = 0.35
    
    # Disegniamo su ENTRAMBI gli assi
    bars_base1 = ax1.bar(x - width/2, square_df['Baseline_GOPS'], width, label='AMD Baseline (32x32x32)', color='#ffb703', edgecolor='black')
    bars_opt1 = ax1.bar(x + width/2, square_df['Opt_GOPS'], width, label='Super-Tuner', color='#219ebc', edgecolor='black')
    
    bars_base2 = ax2.bar(x - width/2, square_df['Baseline_GOPS'], width, color='#ffb703', edgecolor='black')
    bars_opt2 = ax2.bar(x + width/2, square_df['Opt_GOPS'], width, color='#219ebc', edgecolor='black')
    
    threshold = 50
    max_val = max(square_df['Baseline_GOPS'].max(), square_df['Opt_GOPS'].max())
    
    # Asse Superiore
    ax1.set_ylim(threshold, max_val * 1.2) 
    ax1.set_yticks(np.arange(threshold, max_val * 1.25, 250)) 
    
    # Asse Inferiore
    ax2.set_ylim(0, threshold)              
    ax2.set_yticks(np.arange(0, 51, 10)) 
    
    # Nascondiamo i bordi che si toccano
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False, bottom=False) 
    ax2.xaxis.tick_bottom()
    
    # Disegniamo le sbarrette diagonali (//) solo sull'asse Y per far capire il salto di scala
    d = .015  
    kwargs = dict(transform=ax1.transAxes, color='black', clip_on=False, lw=1.5)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  
    kwargs.update(transform=ax2.transAxes)  
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  

    # Etichette asse X
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.set_xlabel('Matrix Dimensions', fontsize=12, fontweight='bold')
    
    # Titoli e legende
    ax1.set_title('Performance on Perfectly Square Matrices (i16)', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=11, loc='upper left')
    
    fig.text(0.03, 0.5, 'Hardware Throughput (GOPS)', va='center', rotation='vertical', fontsize=12, fontweight='bold')
    
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
                        
    for i in range(len(bars_opt1)):
        opt_m = int(square_df.iloc[i]['Opt_m'])
        opt_k = int(square_df.iloc[i]['Opt_k'])
        opt_n = int(square_df.iloc[i]['Opt_n'])
        val = square_df.iloc[i]['Opt_GOPS']
        
        label = f"[{opt_m}x{opt_k}x{opt_n}]"
        
        if val > threshold:
            ax1.annotate(label, xy=(bars_opt1[i].get_x() + bars_opt1[i].get_width() / 2, val),
                         xytext=(0, 6), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#126782')
        elif 0 < val <= threshold:
            ax2.annotate(label, xy=(bars_opt2[i].get_x() + bars_opt2[i].get_width() / 2, val),
                         xytext=(0, 6), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#126782')

    plt.subplots_adjust(left=0.12, bottom=0.15) 
    plt.savefig(f"{output_dir}/02_draft_square_tradeoff.png", bbox_inches='tight')
    plt.close()

# ==============================================================================
# GRAFICO 3: Scatter Plot (Traffico DDR vs Speedup)
# ==============================================================================
def plot_memory_vs_speedup():
    valid = df[(df['Throughput_Speedup'] > 0) & (df['DDR_Traffic_Reduction'] > 0)]
    if valid.empty: return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    scatter = ax.scatter(valid['DDR_Traffic_Reduction'], valid['Throughput_Speedup'], 
                         c=valid['Throughput_Speedup'], cmap='viridis', alpha=0.7, edgecolors='black', s=80)
    
    # Linee di riferimento (Speedup 1x e Riduzione DDR 1x)
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Baseline Throughput')
    ax.axvline(1.0, color='blue', linestyle='--', alpha=0.5, label='Baseline DDR Traffic')
    
    ax.set_xlabel('DDR Traffic Reduction Factor', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput Speedup Factor', fontsize=12, fontweight='bold')
    ax.set_title('Correlation: Saving Memory Bandwidth Boosts Performance', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Speedup (x)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_draft_scatter_memory_speedup.png")
    plt.close()

if __name__ == "__main__":
    plot_scaling_N()
    plot_scaling_K()         
    plot_scaling_M()
    plot_square_matrices()
    # plot_memory_vs_speedup()
    print("[Plotter] Images saved in draft_plots'.")