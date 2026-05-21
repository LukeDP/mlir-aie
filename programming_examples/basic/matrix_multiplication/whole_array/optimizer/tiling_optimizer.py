import math

def get_divisors(n):
    """Utility function to find all divisors of a given integer n."""
    return [i for i in range(1, n + 1) if n % i == 0]

def find_candidates(M, K, N, dtype="bf16", cols=8):
    if dtype == "bf16":
        r, s, t = 4, 8, 8
    else:
        r, s, t = 4, 4, 8
    
    rows = 2  
    
    m_candidates = [m for m in get_divisors(M) if m % r == 0 and M % (m * rows) == 0]
    k_candidates = [k for k in get_divisors(K) if k % s == 0]
    n_candidates = [n for n in get_divisors(N) if n % (2 * t) == 0 and N % (n * cols) == 0]
    
    return m_candidates, k_candidates, n_candidates, rows


def filter_by_memory(m_list, k_list, n_list, M_global, K_global, N_global, dtype="bf16", cols=8):
    """Filter tile combinations based on L1 and L2 memory constraints and volume alignment."""
    valid_combinations = []
    L1_LIMIT = 44 * 1024  
    L2_LIMIT = 512 * 1024 
    
    # rows in base al tipo di dato
    rows = 2
    n_A_tiles_per_shim = max(1, rows // cols) if cols < 4 else 1
    
    c_bytes = 2 if dtype == "bf16" else 4  
    
    for m in m_list:
        for k in k_list:
            for n in n_list:
                # 1. Verifica coerenza geometrica per evitare l'errore di shape in whole_array_poc_heuristic.py
                # Il volume del trasferimento ND-DMA di A non deve eccedere lo spazio allocato (m * K_global)
                iterations_N = N_global // n // cols
                iterations_K = K_global // k
                total_transfer_elements = iterations_N * iterations_K * (m * n_A_tiles_per_shim) * k
                allocated_elements = m * K_global
                
                if total_transfer_elements > allocated_elements:
                    continue  # Scarta questa combinazione per evitare il crash del generatore MLIR
                
                # 2. Controllo dei consumi di memoria
                l1_usage = 2 * ((m * k * 2) + (k * n * 2) + (m * n * 4))
                l2_usage = (m * K_global * 2) + 2 * (k * n * 2) + 2 * (m * n * c_bytes * 2)
                
                if l1_usage <= L1_LIMIT and l2_usage <= L2_LIMIT:
                    if K_global >= 2048 and m > 32:
                        continue
                    valid_combinations.append((m, k, n, l1_usage, l2_usage))
    return valid_combinations

def calculate_score(m, k, n, M, K, N, alpha, gamma, sigma, dtype="bf16"):
    """Heuristic Cost Function (J) to evaluate tiling quality dynamically."""
    tile_ops = 2 * m * n * k
    c_bytes = 2 if dtype == "bf16" else 4
    
    # Calcolo dell'Arithmetic Intensity (AI) basato sui byte reali trasferiti
    tile_bytes = (m * k * 2) + (k * n * 2) + (m * n * c_bytes / (K/k))
    tile_ai = tile_ops / tile_bytes
    
    # NPU Efficiency Term: cambia il divisore di k a seconda dell'architettura vettoriale
    k_div = 64 if dtype == "bf16" else 32
    term_npu = (128/m)**3 + (k_div/k)**2 + (64/n)**2 
    
    # Bandwidth Term
    term_bandwidth = 1 / tile_ai
    
    # Host Setup Term
    num_iterations = (M // m) * (N // (n * 8))
    term_setup = num_iterations * 1.2
    
    # Total Score J
    j = (alpha * term_bandwidth) + (gamma * term_npu) + (sigma * term_setup)
    return j, tile_ai

def solve_mapping(M, K, N, alpha=150.0, gamma=100.0, sigma=30.0, dtype="bf16"): 
    """Main entry point for the tiling optimizer."""
    m_c, k_c, n_c, rows = find_candidates(M, K, N, dtype=dtype)
    # Passiamo M, K, N globali per abilitare il filtro di coerenza volumetrica
    valid_configs = filter_by_memory(m_c, k_c, n_c, M, K, N, dtype=dtype, cols=8)
    
    best_config = None
    min_j = float('inf')
    results = []
    
    for m, k, n, l1, l2 in valid_configs:
        j, ai = calculate_score(m, k, n, M, K, N, alpha, gamma, sigma, dtype=dtype)
        results.append((m, k, n, j, ai))
        if j < min_j:
            min_j = j
            best_config = (m, k, n, j, ai)
            
    return best_config, results