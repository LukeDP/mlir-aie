import os
import matplotlib.pyplot as plt
import numpy as np

def generate_memory_plot(M, K, N, dtype, m, k, n, mode):
    """
    Analytically calculates DDR memory accesses and emits a profile chart.
    Supports both baseline and optimized modes dynamically, writing everything in English.
    """
    in_bytes = 2  # Both bf16 and i16 occupy 2 bytes at input
    c_bytes = 2 if dtype == "bf16" else 4
    
    tiles_M = M // m
    tiles_N = N // n
    tiles_K = K // k
    
    # Analytical traffic calculation based on data stationarity
    if mode == "baseline":
        # AMD standard baseline reloads A for each column block of B
        ddr_reads_A = tiles_M * tiles_K * tiles_N * (m * k * in_bytes)
        title_label = f"AMD Baseline (Standard Tiling: {m}x{k}x{n})"
        color_palette = ['#ff8888', '#ffaaaa', '#ffcccc', '#d11a1a']
    else:
        # Optimized uses Stationary-A: A is read from DDR only once globally
        ddr_reads_A = tiles_M * tiles_K * (m * k * in_bytes)
        title_label = f"Optimized Super-Tuner (Tiling: {m}x{k}x{n})"
        color_palette = ['#66bb66', '#99cc99', '#c2e0c2', '#1e7b1e']
        
    ddr_reads_B = tiles_M * tiles_K * tiles_N * (k * n * in_bytes)
    ddr_writes_C = tiles_M * tiles_N * (m * n * c_bytes)
    total_ddr = ddr_reads_A + ddr_reads_B + ddr_writes_C
    
    # Convert to Megabytes
    to_mb = 1024 * 1024
    categories = ['Matrix A Reads', 'Matrix B Reads', 'Matrix C Writes', 'Total DDR Traffic']
    volumes = [ddr_reads_A / to_mb, ddr_reads_B / to_mb, ddr_writes_C / to_mb, total_ddr / to_mb]
    
    # Create output directory if it doesn't exist
    output_dir = "../memory_profiles"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot construction
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    bars = ax.bar(categories, volumes, color=color_palette, edgecolor='black', width=0.5)
    
    ax.set_ylabel('Data Transferred Volume (Megabytes)', fontsize=12, fontweight='bold')
    ax.set_title(f'External Memory Access Profile (DDR)\nGlobal Shape: {M}x{K}x{N} ({dtype}) - {title_label}', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f} MB',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    # Calculate local Arithmetic Intensity
    tile_ops = 2 * m * n * k
    tile_bytes = (m * k * in_bytes) + (k * n * in_bytes) + (m * n * c_bytes)
    ai = tile_ops / tile_bytes
    
    # Info Box
    textstr = f"Mode: {mode.upper()}\nTiling: {m}x{k}x{n}\nArithmetic Intensity: {ai:.2f} OP/Byte"
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8, edgecolor='gray')
    ax.text(0.05, 0.92, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    
    fig.tight_layout()
    
    # Save the plot inside the dedicated folder
    output_name = f"{output_dir}/memory_profile_{mode}_{M}x{K}x{N}_{dtype}.png"
    plt.savefig(output_name)
    plt.close()
    print(f"[Plotter] Memory profile chart saved for {mode.upper()}: {output_name}")