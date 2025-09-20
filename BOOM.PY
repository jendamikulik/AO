import numpy as np
import argparse
from math import gcd, ceil, log2

def generate_hadamard(n):
    """Generate Sylvester-Hadamard matrix of size n x n where n is power of 2."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("n must be a power of 2")
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H

def find_coprime(n, start=3):
    """Find a number coprime to n, starting from start."""
    for k in range(start, n):
        if gcd(k, n) == 1:
            return k
    return 1

def power_method(A, num_iter=200, seed=None):
    """Power method to estimate largest eigenvalue (real for Hermitian)."""
    if seed is not None:
        np.random.seed(seed)
    n = A.shape[0]
    v = np.random.randn(n) + 1j * np.random.randn(n)
    v /= np.linalg.norm(v)
    for _ in range(num_iter):
        v = A @ v
        v /= np.linalg.norm(v)
    lambda_max = np.real(v.conj().T @ A @ v / (v.conj().T @ v))
    return lambda_max

def get_neighbors(i, C, deg):
    """Get ring neighbors for i."""
    half = deg // 2
    neighbors = [(i + k) % C for k in range(-half, half + 1) if k != 0]
    return neighbors

def compute_s2_metrics(G, C, deg):
    """Compute S2 metrics: |G_ij| max/avg off-diag, row-sum neighbors max/avg."""
    off_diag_abs = np.abs(G.copy())
    np.fill_diagonal(off_diag_abs, 0)
    max_Gij = np.max(off_diag_abs)
    avg_Gij = np.sum(off_diag_abs) / (C * (C - 1))
    row_sums = []
    for i in range(C):
        neigh = get_neighbors(i, C, deg)
        row_sum = np.sum(off_diag_abs[i, neigh])
        row_sums.append(row_sum)
    max_row_sum = max(row_sums)
    avg_row_sum = np.mean(row_sums)
    return max_Gij, avg_Gij, max_row_sum, avg_row_sum

def main(args):
    C = args.C
    cR = args.cR
    T = C * cR
    rho_lock = args.rho_lock
    zeta0 = args.zeta0
    deg = args.deg
    mode = args.mode
    seed = args.seed
    tau = args.tau

    if seed is not None:
        np.random.seed(seed)

    # Compute m
    m = int(rho_lock * T)

    # Hadamard size
    log_m = ceil(log2(m))
    H_len = 2 ** log_m
    H = generate_hadamard(H_len)

    # Strides (deterministic)
    row_stride = find_coprime(H_len, start=3)
    col_stride = find_coprime(H_len, start=5)

    # Offsets
    s = T // 2
    while gcd(s, T) != 1 and s > 1:
        s -= 1
    offsets = np.array([(j * s) % T for j in range(C)])

    # Mode-specific settings
    fixed_row = 1
    fixed_col_start = 0
    fixed_offset = 0
    adversarial_fraction = 0.2
    adversarial_interval = int(1 / adversarial_fraction)  # every 5th

    # Phase matrix
    Phi = np.full((T, C), np.pi)  # default pi outside lock

    for j in range(C):
        if mode == 'sat':
            row_j = fixed_row
            col_start = fixed_col_start
            lock_start = fixed_offset
        else:
            row_j = (j * row_stride) % H_len
            col_start = j % H_len
            lock_start = offsets[j]
            if mode == 'unsat_hadamard' and j % adversarial_interval == 0:
                row_j = -1  # adversarial

        # Subsampled indices
        sub_indices = []
        current = col_start
        for _ in range(m):
            sub_indices.append(current)
            current = (current + col_stride) % H_len
        sub_indices = np.array(sub_indices)

        # Mask
        if row_j == -1:
            mask = -np.ones(m)
        else:
            mask = H[row_j, sub_indices]

        # Negative positions
        negative_ks = [k for k in range(m) if mask[k] == -1]

        # Select kk positions from negatives
        kk = int(zeta0 * m)
        selected_ks = negative_ks[:kk]  # deterministic selection

        # Lock positions (handle wrap-around)
        lock_pos = [(lock_start + k) % T for k in range(m)]

        # Set in lock
        for k in range(m):
            if k in selected_ks:
                Phi[lock_pos[k], j] = np.pi
            else:
                Phi[lock_pos[k], j] = 0.0

    # Z matrix
    Z = np.exp(1j * Phi)

    # Gram matrix (Hermitian, no abs)
    G = (1 / T) * Z.conj().T @ Z

    # lambda_max via power method
    lambda_max = power_method(G, seed=seed)

    # mu
    mu = lambda_max / C

    # Decision
    decision = "SAT" if mu > tau else "UNSAT"

    # Lock-only Z
    Z_lock = np.zeros_like(Z)
    for j in range(C):
        if mode == 'sat':
            lock_start = fixed_offset
        else:
            lock_start = offsets[j]
        lock_pos = [(lock_start + k) % T for k in range(m)]
        for k in range(m):
            Z_lock[lock_pos[k], j] = Z[lock_pos[k], j]

    # Lock-only Gram
    G_lock = (1 / m) * Z_lock.conj().T @ Z_lock

    # S2 metrics
    all_t_max_Gij, all_t_avg_Gij, all_t_max_row, all_t_avg_row = compute_s2_metrics(G, C, deg)
    lock_max_Gij, lock_avg_Gij, lock_max_row, lock_avg_row = compute_s2_metrics(G_lock, C, deg)

    # Theoretical kappa_S2
    kappa_S2 = (1 - 2 * zeta0) ** 2 + 2 ** (-log_m / 2) + 2 / m + 1 / T
    d_kappa_S2 = deg * kappa_S2

    # Output
    print(f"Mode: {mode}")
    print(f"Parameters: C={C}, T={T}, m={m}, zeta0={zeta0}, deg={deg}")
    print(f"mu: {mu:.4f}")
    print(f"Decision (tau={tau}): {decision}")
    print("\nAll-T S2 Metrics:")
    print(f"  |G_ij| max: {all_t_max_Gij:.4f}, avg: {all_t_avg_Gij:.4f}")
    print(f"  Row-sum neighbors max: {all_t_max_row:.4f}, avg: {all_t_avg_row:.4f}")
    print("\nLock-only S2 Metrics:")
    print(f"  |G_ij| max: {lock_max_Gij:.4f}, avg: {lock_avg_Gij:.4f}")
    print(f"  Row-sum neighbors max: {lock_max_row:.4f}, avg: {lock_avg_row:.4f}")
    print(f"\nTheoretical kappa_S2: {kappa_S2:.4f}, d * kappa_S2: {d_kappa_S2:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic Spectral Tester (LST-DEC)")
    parser.add_argument("--mode", type=str, required=True, choices=["sat", "unsat", "unsat_hadamard"])
    parser.add_argument("--C", type=int, required=True)
    parser.add_argument("--cR", type=int, required=True)
    parser.add_argument("--rho_lock", type=float, required=True)
    parser.add_argument("--zeta0", type=float, required=True)
    parser.add_argument("--deg", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=0.80)
    args = parser.parse_args()
    main(args)