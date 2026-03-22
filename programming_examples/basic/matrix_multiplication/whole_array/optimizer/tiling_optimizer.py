import math

def get_divisors(n):
    """Restituisce tutti i divisori di un numero n."""
    return [i for i in range(1, n + 1) if n % i == 0]

def find_candidates(M, K, N, cols=8, rows=4):
    """Trova tutti i possibili m, k, n che rispettano i vincoli di divisibilità e MAC."""
    
    # Vincoli microkernel MAC (i16 su npu2)
    r, s, t = 4, 4, 8
    
    # m deve essere tale che (m * rows) divida M E m sia multiplo di r
    m_candidates = [m for m in get_divisors(M) if m % r == 0 and M % (m * rows) == 0]
    
    # k deve dividere K E essere multiplo di s
    k_candidates = [k for k in get_divisors(K) if k % s == 0]
    
    # n deve essere tale che (n * cols) divida N E n sia multiplo di t
    n_candidates = [n for n in get_divisors(N) if n % t == 0 and N % (n * cols) == 0]
    
    return m_candidates, k_candidates, n_candidates

def filter_by_memory(m_list, k_list, n_list, K_global):
    valid_combinations = []
    
    L1_LIMIT = 32 * 1024  # 32 KB
    L2_LIMIT = 512 * 1024 # 512 KB
    
    for m in m_list:
        for k in k_list:
            for n in n_list:
                # Calcolo L1 (Double Buffering)
                # A: i16 (2B), B: i16 (2B), C: i32 (4B)
                l1_usage = 2 * ((m * k * 2) + (k * n * 2) + (m * n * 4))
                
                # Calcolo L2 (Modello Stationary-A PoC)
                # A è residente per tutta la dimensione K
                # B e C sono in ping-pong (C è accumulato per 4 righe)
                l2_usage = (m * K_global * 2) + 2 * (k * n * 2) + 2 * (m * n * 4 * 4)
                
                if l1_usage <= L1_LIMIT and l2_usage <= L2_LIMIT:
                    valid_combinations.append((m, k, n, l1_usage, l2_usage))
                    
    return valid_combinations

def calculate_score(m, k, n, M, K, N, alpha=1.0, gamma=0.5, sigma=2.0):
    dsize = 2 # int16
    
    # 1. Calcolo Arithmetic Intensity (AI)
    ops = 2 * M * K * N
    # Formula PoC (Stationary-A: Matrix A letta una sola volta)
    total_bytes = (M * K + K * N + M * N) * dsize
    ai = ops / total_bytes
    
    # 2. Termine Banda: vogliamo AI alto, quindi 1/AI basso
    term_bandwidth = 1 / ai
    
    # 3. Termine Contesa (Stall Penalty)
    # Più n è grande, meno fette facciamo, meno contesa abbiamo
    term_contention = 1 / n
    
    # 4. Termine Overhead (Setup Penalty)
    # Calcoliamo quanti "attivazioni" DMA servono. 
    # M/m sono le iterazioni esterne. Più sono, più l'overhead pesa.
    num_iterations = (M // m) * (N // (n * 8)) # 8 è n_cols
    term_overhead = num_iterations * 0.007 # 7ms di setup
    
    # Punteggio finale J (Minore è meglio)
    j = (alpha * term_bandwidth) + (gamma * term_contention) + (sigma * term_overhead)
    
    return j, ai



def solve_mapping(M, K, N):
    m_c, k_c, n_c = find_candidates(M, K, N)
    valid_configs = filter_by_memory(m_c, k_c, n_c, K)
    
    best_config = None
    min_j = float('inf')
    
    results = []
    
    for m, k, n, l1, l2 in valid_configs:
        j, ai = calculate_score(m, k, n, M, K, N)
        results.append((m, k, n, j, ai))
        
        if j < min_j:
            min_j = j
            best_config = (m, k, n, j, ai)
            
    return best_config, results
