import math

def get_divisors(n):
    """Utility function to find all divisors of a given integer n."""
    return [i for i in range(1, n + 1) if n % i == 0]

def find_candidates(M, K, N, cols=8, rows=4):
    """
    Generate potential tiling candidates (m, k, n) based on hardware constraints.
    - r, s, t: microkernel dimensions for i16/i32 on NPU2 (4x4x8).
    - n must be a multiple of 16 (2*t) to satisfy vector alignment requirements.
    """
    r, s, t = 4, 4, 8
    
    # m must be a multiple of r and satisfy the grid distribution (M / (m * rows))
    m_candidates = [m for m in get_divisors(M) if m % r == 0 and M % (m * rows) == 0]
    
    # k must be a multiple of s
    k_candidates = [k for k in get_divisors(K) if k % s == 0]
    
    # n must be a multiple of 16 (2*t) for optimal SIMD performance and alignment
    n_candidates = [n for n in get_divisors(N) if n % (2 * t) == 0 and N % (n * cols) == 0]
    
    return m_candidates, k_candidates, n_candidates

def filter_by_memory(m_list, k_list, n_list, K_global):
    """
    Filter tile combinations based on L1 and L2 memory constraints.
    - L1_LIMIT (44KB): The 'Magic Threshold' that ensures stability by leaving 
      ~20KB for the Peano compiler stack and local variables.
    - Double buffering is accounted for (factor of 2).
    """
    valid_combinations = []
    L1_LIMIT = 44 * 1024  
    L2_LIMIT = 512 * 1024 
    
    for m in m_list:
        for k in k_list:
            for n in n_list:
                # L1 usage: 2 * (A_tile + B_tile + C_tile)
                l1_usage = 2 * ((m * k * 2) + (k * n * 2) + (m * n * 4))
                
                # L2 usage: A_row + 2*(B_tile + C_accumulated)
                l2_usage = (m * K_global * 2) + 2 * (k * n * 2) + 2 * (m * n * 4 * 4)
                
                if l1_usage <= L1_LIMIT and l2_usage <= L2_LIMIT:
                    valid_combinations.append((m, k, n, l1_usage, l2_usage))
    return valid_combinations

def calculate_score(m, k, n, M, K, N, alpha=1.0, gamma=0.5, sigma=2.0):
    """
    Heuristic Cost Function (J) to evaluate tiling quality.
    The goal is to minimize J[cite: 19].
    
    Components:
    1. Bandwidth (alpha): Inverse of Arithmetic Intensity.
    2. NPU Efficiency (gamma): Penalizes small tiles. m < 64 is cubicly penalized 
       to reduce Host jitter and maximize NPU throughput.
    3. Host Overhead (sigma): Based on total setup iterations (task packets).
    """
    # Arithmetic Intensity (AI) calculation
    tile_ops = 2 * m * n * k
    tile_bytes = (m * k * 2) + (k * n * 2) + (m * n * 8 / (K/k))
    tile_ai = tile_ops / tile_bytes
    
    # NPU Efficiency Term: favors larger m, k, n. 
    # Penalty for m is aggressive to stabilize Host execution.
    term_npu = (128/m)**3 + (64/k)**2 + (64/n)**2 
    
    # Bandwidth Term
    term_bandwidth = 1 / tile_ai
    
    # Host Setup Term: total number of command packets sent by the CPU
    num_iterations = (M // m) * (N // (n * 8))
    term_setup = num_iterations * 1.2
    
    # Total Score J
    j = (alpha * term_bandwidth) + (gamma * term_npu) + (sigma * term_setup)
    return j, tile_ai

def solve_mapping(M, K, N, alpha=100.0, gamma=50.0, sigma=1.0): 
    """
    Main entry point for the tiling optimizer.
    Finds the (m, k, n) configuration that minimizes the heuristic score J.
    """
    # 1. Generate all possible hardware-compliant candidates
    m_c, k_c, n_c = find_candidates(M, K, N)
    
    # 2. Filter by memory safety limits
    valid_configs = filter_by_memory(m_c, k_c, n_c, K)
    
    best_config = None
    min_j = float('inf')
    results = []
    
    # 3. Score each valid configuration
    for m, k, n, l1, l2 in valid_configs:
        j, ai = calculate_score(m, k, n, M, K, N, alpha, gamma, sigma)
        results.append((m, k, n, j, ai))
        if j < min_j:
            min_j = j
            best_config = (m, k, n, j, ai)
            
    return best_config, results