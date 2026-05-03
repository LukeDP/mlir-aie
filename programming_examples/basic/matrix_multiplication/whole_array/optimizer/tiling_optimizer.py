import math

def get_divisors(n):
    return [i for i in range(1, n + 1) if n % i == 0]

def find_candidates(M, K, N, cols=8, rows=4):
    r, s, t = 4, 4, 8

    m_candidates = [m for m in get_divisors(M) if m % r == 0 and M % (m * rows) == 0]
    k_candidates = [k for k in get_divisors(K) if k % s == 0]
    n_candidates = [n for n in get_divisors(N) if n % t == 0 and N % (n * cols) == 0]
    
    return m_candidates, k_candidates, n_candidates


def filter_by_memory(m_list, k_list, n_list, K_global):
    valid_combinations = []
    # Aumentiamo a 60KB per permettere tile competitivi lasciando spazio allo stack
    L1_LIMIT = 60 * 1024  
    L2_LIMIT = 512 * 1024 
    
    for m in m_list:
        for k in k_list:
            for n in n_list:
                l1_usage = 2 * ((m * k * 2) + (k * n * 2) + (m * n * 4))
                l2_usage = (m * K_global * 2) + 2 * (k * n * 2) + 2 * (m * n * 4 * 4)
                
                if l1_usage <= L1_LIMIT and l2_usage <= L2_LIMIT:
                    valid_combinations.append((m, k, n, l1_usage, l2_usage))
    return valid_combinations

def calculate_score(m, k, n, M, K, N, alpha=1.0, gamma=0.5, sigma=2.0):
    tile_ops = 2 * m * n * k
    tile_bytes = (m * k * 2) + (k * n * 2) + (m * n * 8 / (K/k))
    tile_ai = tile_ops / tile_bytes
    
    # Efficienza Hardware: Penalizziamo k < 64 E n < 32[cite: 23]
    term_k_efficiency = (128 / k) ** 2 # Preferiamo k=128 per ridurre overhead DMA
    term_m_efficiency = (64 / m) ** 2
    term_n_efficiency = (64 / n) ** 2
    
    term_bandwidth = 1 / tile_ai
    term_contention = 1 / n
    # Aumentiamo l'impatto del setup dell'Host per favorire tile più grandi (m=64)[cite: 23, 33]
    num_iterations = (M // m) * (N // (n * 8))
    term_setup = num_iterations * 0.1 # Aumentato per evitare m=32
    
    
    # Sommiamo tutte le inefficienze hardware[cite: 18]
    j = (alpha * (term_bandwidth + term_k_efficiency + term_n_efficiency + term_m_efficiency)) + \
        (gamma * term_contention) + (sigma * term_setup)
    return j, tile_ai



def solve_mapping(M, K, N, alpha=50.0, gamma=0.1, sigma=100.0): # Pesi suggeriti
    m_c, k_c, n_c = find_candidates(M, K, N)
    valid_configs = filter_by_memory(m_c, k_c, n_c, K)
    
    best_config = None
    min_j = float('inf')
    results = []
    
    for m, k, n, l1, l2 in valid_configs:
        # CORREZIONE: Passiamo alpha, gamma, sigma qui[cite: 28]
        j, ai = calculate_score(m, k, n, M, K, N, alpha, gamma, sigma)
        results.append((m, k, n, j, ai))
        
        if j < min_j:
            min_j = j
            best_config = (m, k, n, j, ai)
            
    return best_config, results

