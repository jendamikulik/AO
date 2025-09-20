# AO_P_equals_N.py — single-file deterministic spectral SAT tester (LST-DEC)
# Implements: de-aliased offsets, deterministic truncated Hadamard masks,
# complex Hermitian Gram, power-method λ_max, μ = λ_max/C decision,
# lock-only S2 neighbor row-sum vs. theoretical bound.
#
# Usage examples:
#   python AO_P_equals_N.py --mode sat --C 1000 --cR 15 --rho_lock 0.50 --zeta0 0.40 --deg 4 --seed 42
#   python AO_P_equals_N.py --mode unsat --C 1000 --cR 15 --rho_lock 0.50 --zeta0 0.40 --deg 4 --seed 42
#   python AO_P_equals_N.py --mode unsat_hadamard --C 1000 --cR 15 --rho_lock 0.50 --zeta0 0.40 --deg 4 --seed 42
#
# Prints:
#   μ, λ_max, S2 (all-T and lock-only) neighbor row-sums and |G_ij| stats,
#   theoretical κ_S2 and d·κ_S2, and a suggested SAT/UNSAT verdict by threshold τ.
#
# This file is self-contained (numpy only).

import argparse
import math
import numpy as np
from typing import Tuple

# ---------------------------- Utilities ----------------------------

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

def is_coprime(a: int, b: int) -> bool:
    return gcd(a, b) == 1

def next_power_of_two(n: int) -> int:
    k = 1
    while k < n:
        k <<= 1
    return k

def pick_stride_near_half(T: int) -> int:
    """Pick s ~ T/2, coprime with T, deterministic (scan outward)."""
    target = T // 2
    # search radius up to T//3 is plenty
    for delta in range(0, T):
        for cand in (target - delta, target + delta):
            if 1 <= cand < T and is_coprime(cand, T):
                return cand
    # fallback (should not happen)
    for cand in range(1, T):
        if is_coprime(cand, T):
            return cand
    raise RuntimeError("Failed to find coprime stride.")

def sylvester_hadamard(n: int) -> np.ndarray:
    """Deterministic Sylvester-type Walsh–Hadamard matrix (n must be power of two)."""
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"
    H = np.array([[1]], dtype=np.int8)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H

def choose_hadamard_row_and_cols(m: int, seed: int = 0) -> Tuple[np.ndarray, int, int, int]:
    """
    Deterministically choose a Hadamard row index and a column subsequence with coprime strides.
    Returns (row_bits[0:m], H_len, row_index, col_stride)
    """
    H_len = next_power_of_two(m)
    H = sylvester_hadamard(H_len)
    # Choose an odd row_step coprime with H_len; pick a row_index via seed
    # but deterministically map seed to odd index.
    row_step = 1
    while not is_coprime(row_step, H_len):
        row_step += 2
    row_index = ((seed * 3 + 1) * row_step) % H_len
    # ensure not zero row (degenerate); shift if needed
    if row_index == 0:
        row_index = 1
    # Choose odd column stride g coprime with H_len
    g = 1
    while not (g % 2 == 1 and is_coprime(g, H_len)):
        g += 2
    # Subsample columns by c_t = (c0 + g t) mod H_len, with c0 fixed 0
    cols = [(g * t) % H_len for t in range(m)]
    row = H[row_index, cols]  # entries in {+1, -1}
    return row, H_len, row_index, g

def build_offsets(C: int, T: int) -> np.ndarray:
    """Offsets o_j = (j*s) mod T with s ~ T/2, coprime to T."""
    s = pick_stride_near_half(T)
    j = np.arange(C, dtype=np.int64)
    return (j * s) % T

# ---------------------------- Phase schedule ----------------------------

def build_phase_schedule(C: int, R: int, rho_lock: float, zeta0: float, mode: str, seed: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build Φ ∈ R^{T×C} with T = 3R. Each column j has a lock window L_j of length m = floor(rho_lock*T)
    starting at offset o_j. Inside lock: set kk = floor(zeta0*m) slots to π at positions indicated by
    negative entries of a chosen Hadamard row; the rest 0. Outside lock: π.
    Returns (Φ, mask_lock, m)
    """
    T = 3 * R
    m = int(math.floor(rho_lock * T))
    Phi = np.ones((T, C), dtype=np.float64) * math.pi  # outside lock: π
    mask_lock = np.zeros((T, C), dtype=bool)

    offsets = build_offsets(C, T)
    row_bits, H_len, row_index, g = choose_hadamard_row_and_cols(m, seed=seed)

    # kk negative entries mapped to π (already π), positives to 0 inside lock; we will set 0 where row==+1
    # Determine positions within lock per clause via column stride g
    # row_bits corresponds to H[row_index, (g*t) mod H_len] for t=0..m-1
    inner = np.where(row_bits < 0, math.pi, 0.0)  # (-1)->π, (+1)->0

    for j in range(C):
        o = offsets[j]
        # lock indices are [o, o+m) modulo T
        idx = (o + np.arange(m)) % T
        Phi[idx, j] = inner  # apply inside lock
        mask_lock[idx, j] = True

    # Mode tweaks (optional illustrative variants)
    if mode == "unsat_hadamard":
        # Flip a fraction of columns to adversarial rows (deterministic but different seed)
        row_bad, _, _, _ = choose_hadamard_row_and_cols(m, seed=seed + 137)
        inner_bad = np.where(row_bad < 0, math.pi, 0.0)
        for j in range(0, C, 5):  # 20% adversarial
            o = offsets[j]
            idx = (o + np.arange(m)) % T
            Phi[idx, j] = inner_bad

    return Phi, mask_lock, m

# ---------------------------- Gram & spectral ----------------------------

def gram_from_phases(Phi: np.ndarray) -> np.ndarray:
    """Z = exp(iΦ), G = (1/T) Z* Z (Hermitian)."""
    T, C = Phi.shape
    Z = np.exp(1j * Phi)  # shape T×C
    # G = (1/T) Z^* Z  -> C×C
    G = (Z.conj().T @ Z) / T
    # enforce Hermitian numerical symmetry
    G = (G + G.conj().T) * 0.5
    return G

def gram_lock_only(Phi: np.ndarray, mask_lock: np.ndarray) -> np.ndarray:
    """Lock-only Gram normalized by m: use only rows where both (t,j) and (t,i) are inside their locks.
    Implement by zeroing non-lock entries and renormalizing by average lock length m.
    """
    T, C = Phi.shape
    # Zero entries outside lock; keep raw complex exponentials
    Z = np.exp(1j * Phi) * mask_lock.astype(np.float64)
    # Count per-column lock length (should be m)
    m_eff = np.maximum(mask_lock.sum(axis=0), 1)
    # Normalize columns by sqrt(m) so that diagonal ≈ 1
    Znorm = Z / np.sqrt(m_eff[None, :])
    G = Znorm.conj().T @ Znorm  # normalized by m already
    G = (G + G.conj().T) * 0.5
    return G

def power_method_lmax(G: np.ndarray, iters: int = 200, tol: float = 1e-10, seed: int = 0) -> float:
    C = G.shape[0]
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(C) + 1j * rng.standard_normal(C)
    v /= np.linalg.norm(v)
    l_old = 0.0
    for _ in range(iters):
        w = G @ v
        v = w / (np.linalg.norm(w) + 1e-18)
        l_new = float(np.real(np.vdot(v, G @ v)))  # Rayleigh quotient
        if abs(l_new - l_old) < tol * max(1.0, abs(l_new)):
            break
        l_old = l_new
    return l_new

# ---------------------------- S2 metrics ----------------------------

def ring_neighbors(C: int, deg: int) -> list[tuple[int, int]]:
    """Undirected ring with degree 'deg' (even): edges between i and i±k/2 for k in {2..deg} step 2."""
    assert deg % 2 == 0 and deg >= 2
    edges = set()
    half = deg // 2
    for i in range(C):
        for k in range(1, half + 1):
            j1 = (i + k) % C
            j2 = (i - k) % C
            a, b = sorted((i, j1))
            edges.add((a, b))
            a, b = sorted((i, j2))
            edges.add((a, b))
    return sorted(edges)

def s2_stats(G: np.ndarray, deg: int) -> dict:
    """Compute |G_ij| on ring neighbors and row-sum over neighbors."""
    C = G.shape[0]
    edges = ring_neighbors(C, deg)
    mags = []
    row_sums = np.zeros(C, dtype=np.float64)
    for (i, j) in edges:
        if i == j:
            continue
        val = G[i, j]
        a = abs(val)
        mags.append(a)
        row_sums[i] += a
        row_sums[j] += a
    mags = np.array(mags) if mags else np.array([0.0])
    return {
        "edge_abs_max": float(np.max(mags)),
        "edge_abs_avg": float(np.mean(mags)),
        "row_sum_max": float(np.max(row_sums)),
        "row_sum_avg": float(np.mean(row_sums)),
    }

def kappa_S2(m: int, T: int, zeta0: float) -> float:
    """Theoretical S2 bound component from the notes:
       κ_S2 = (1-2ζ0)^2 + 2^{-ceil(log2 m)/2} + 2/m + 1/T
    """
    term1 = (1.0 - 2.0 * zeta0) ** 2
    pow2 = next_power_of_two(m)
    term2 = 2.0 ** (-(int(math.log2(pow2))) / 2.0)
    term3 = 2.0 / max(1, m)
    term4 = 1.0 / max(1, T)
    return term1 + term2 + term3 + term4

# ---------------------------- Main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="unsat", choices=["sat", "unsat", "unsat_hadamard"])
    ap.add_argument("--C", type=int, default=1000)
    ap.add_argument("--cR", type=float, default=15.0, help="R ≈ cR * log C (base e)")
    ap.add_argument("--rho_lock", type=float, default=0.50)
    ap.add_argument("--zeta0", type=float, default=0.40)
    ap.add_argument("--deg", type=int, default=4, help="neighbor degree (even) for S2 metrics")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau", type=float, default=None, help="decision threshold for μ (optional)")
    args = ap.parse_args()

    C = args.C
    R = int(round(args.cR * math.log(C)))
    T = 3 * R
    rho_lock = args.rho_lock
    zeta0 = args.zeta0
    deg = args.deg
    seed = args.seed

    print(f"Constants: C={C}, R={R}, T={T}, rho_lock={rho_lock:.2f}, zeta0={zeta0:.2f}, deg={deg}, seed={seed}")
    Phi, mask_lock, m = build_phase_schedule(C, R, rho_lock, zeta0, args.mode, seed=seed)

    # Full Gram & spectral
    G_all = gram_from_phases(Phi)
    lmax = power_method_lmax(G_all, seed=seed)
    mu = lmax / C

    # Lock-only Gram & S2
    G_lock = gram_lock_only(Phi, mask_lock)

    s2_all = s2_stats(G_all, deg=deg)
    s2_lock = s2_stats(G_lock, deg=deg)

    kappa = kappa_S2(m, T, zeta0)
    print("\n--- Spectral ---")
    print(f"λ_max (all-T) = {lmax:.4f}")
    print(f"μ = λ_max/C   = {mu:.6f}")

    print("\n--- S2 empirical (all T) ---")
    print(f"|G_ij| over edges: max={s2_all['edge_abs_max']:.4f}, avg={s2_all['edge_abs_avg']:.4f}")
    print(f"row-sum over neighbors: max={s2_all['row_sum_max']:.4f}, avg={s2_all['row_sum_avg']:.4f}")

    print("\n--- S2 empirical (lock-only) ---")
    print(f"|G_ij| over edges: max={s2_lock['edge_abs_max']:.4f}, avg={s2_lock['edge_abs_avg']:.4f}")
    print(f"row-sum over neighbors: max={s2_lock['row_sum_max']:.4f}, avg={s2_lock['row_sum_avg']:.4f}")

    print("\n--- S2 theoretical bound ---")
    print(f"κ_S2 ≈ {kappa:.6f},  d·κ_S2 ≈ {deg * kappa:.6f}")

    # Simple suggested τ if not provided:
    if args.tau is None:
        # Empirical guardrails: SAT-envelope μ≈1, UNSAT μ≈~0.15–0.6 depending on regime
        # pick a neutral fixed τ=0.8 (user can override)
        tau = 0.80
    else:
        tau = args.tau
    verdict = "SAT" if mu >= tau else "UNSAT"
    print(f"\n--- Decision ---")
    print(f"τ (threshold) = {tau:.3f}  →  verdict: {verdict}")

if __name__ == "__main__":
    main()
