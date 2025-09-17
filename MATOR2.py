#!/usr/bin/env python3
# AO_FINALE.py — deterministic time-averaged spectral tester + S2 + holonomy demo
# (de-aliased offsets, Hadamard masks, complex Hermitian Gram, lock-only S2)

import argparse, math, numpy as np
from numpy.linalg import eigh

# ------------------------------- Utilities -------------------------------

def hadamard(n: int) -> np.ndarray:
    """Walsh–Hadamard matrix (size is next power of two)."""
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H],[H, -H]])
    return H

def next_pow2(n: int) -> int:
    k = 1
    while k < n: k <<= 1
    return k

def angles_mod(x):
    """Wrap to (-pi, pi]."""
    return (x + np.pi) % (2*np.pi) - np.pi

# -------------------------- Schedule construction ------------------------

def stride_near_half_coprime(T: int) -> int:
    """Pick s ≈ T/2 - 1, coprime with T."""
    s = max(1, T//2 - 1)
    while math.gcd(s, T) != 1:
        s -= 1
        if s <= 0:
            s = 1
            break
    return s

def schedule_sat_envelope(C: int, R: int, L: int = 3) -> np.ndarray:
    """SAT envelope: all phases aligned to 0 on [0, T)."""
    T = R * L
    return np.zeros((T, C), dtype=float)

def schedule_unsat_hadamard(
    C: int, R: int, rho_lock: float = 0.5, zeta0: float = 0.4, L: int = 3,
    row_step: int | None = None, col_stride: int | None = None, col_offset: int = 0,
    s_stride: int | None = None, seed: int = 42, return_locks: bool = True
):
    """
    Deterministic UNSAT-like schedule:
      • offsets: de-aliased stride near T/2, coprime with T
      • inside lock: Hadamard row/column with coprime strides
      • k = floor(zeta0*m) of the lock slots set to π (negatives), rest 0
      • outside lock: π
    Returns Phi (T×C) and lock indices per clause.
    """
    rng = np.random.default_rng(seed)
    T = R * L
    m = int(round(rho_lock * T))
    k = int(round(zeta0 * m))

    # offsets
    s = s_stride if s_stride is not None else stride_near_half_coprime(T)
    if math.gcd(s, T) != 1:
        raise ValueError("s_stride must be coprime with T")
    offsets = [(j * s) % T for j in range(C)]
    locks = [np.array([(offsets[j] + t) % T for t in range(m)], dtype=int) for j in range(C)]

    # base
    Phi = np.full((T, C), np.pi, dtype=float)

    # Hadamard config
    Hlen = next_pow2(m)
    H = hadamard(Hlen)

    # row stride (coprime with Hlen, large & odd)
    if row_step is None:
        row_step = (Hlen // 2) + 1
    if math.gcd(row_step, Hlen) != 1:
        row_step |= 1
        while math.gcd(row_step, Hlen) != 1:
            row_step += 2

    # column stride (coprime with Hlen, odd)
    if col_stride is None:
        g = (Hlen // 3) | 1
    else:
        g = int(col_stride) | 1
    while math.gcd(g, Hlen) != 1:
        g += 2
    cols = (int(col_offset) + g * np.arange(m)) % Hlen

    # fill lock slots
    for j in range(C):
        row = H[(j * row_step) % Hlen, cols]
        neg = np.flatnonzero(row < 0.0)
        if len(neg) >= k:
            mask_pi = rng.choice(neg, size=k, replace=False)
        else:
            extra = rng.choice(np.setdiff1d(np.arange(m), neg), size=k - len(neg), replace=False)
            mask_pi = np.concatenate([neg, extra])
        mask_0 = np.setdiff1d(np.arange(m), mask_pi)
        slots = locks[j]
        Phi[slots[mask_pi], j] = np.pi
        Phi[slots[mask_0], j] = 0.0

    return (Phi, locks) if return_locks else Phi

# ---------------------------- Spectral witness ---------------------------

def gram_complex(Phi: np.ndarray) -> np.ndarray:
    """Complex Hermitian Gram G = (1/T) Z* Z, diagonal set to 1."""
    Z = np.exp(1j * Phi)
    G = (Z.conj().T @ Z) / Phi.shape[0]
    G = 0.5 * (G + G.conj().T)
    np.fill_diagonal(G, 1.0 + 0j)
    return G

def top_mu_lambda(G: np.ndarray) -> tuple[float, float]:
    """Return (mu, lambda_max)."""
    evals = eigh(G, UPLO='U')[0]
    lam = float(evals[-1])
    C = G.shape[0]
    return lam / C, lam

# ------------------------------ S2 metrics -------------------------------

def gram_lock_only(Phi: np.ndarray, locks: list[np.ndarray]) -> np.ndarray:
    """
    Lock-only Gram with normalization by m (NOT by |intersection|).
    """
    T, C = Phi.shape
    Z = np.exp(1j * Phi)
    m = len(locks[0])
    G = np.zeros((C, C), dtype=np.complex128)
    for i in range(C):
        Li = set(locks[i].tolist())
        for j in range(i, C):
            Lj = set(locks[j].tolist())
            inter = np.array(sorted(Li & Lj), dtype=int)
            if inter.size == 0:
                val = 0.0
            else:
                val = (Z[inter, i].conj() * Z[inter, j]).sum() / m
            G[i, j] = val
            G[j, i] = np.conjugate(val)
    np.fill_diagonal(G, 1.0 + 0j)
    return G

def edge_stats(G: np.ndarray, neighbors: list[set[int]]) -> dict:
    """Per-edge |G_ij| stats and row-sum over neighbor sets."""
    C = G.shape[0]
    edges = []
    row_sums = np.zeros(C, dtype=float)
    A = np.abs(G)
    for i in range(C):
        s = 0.0
        for j in neighbors[i]:
            if j > i:
                edges.append(A[i, j])
            s += A[i, j]
        row_sums[i] = s
    edges = np.array(edges) if edges else np.array([0.0])
    return dict(
        max_edge=float(edges.max()),
        avg_edge=float(edges.mean()),
        max_row_sum=float(row_sums.max()),
        avg_row_sum=float(row_sums.mean()),
    )

def kappa_S2(m: int, T: int, zeta0: float) -> float:
    """κ_S2 = (1-2ζ0)^2 + 2^{-ceil(log2 m)/2} + 2/m + 1/T."""
    term1 = (1.0 - 2.0 * zeta0) ** 2
    pow2 = 1 << (int(np.ceil(np.log2(max(1, m)))))
    term2 = (pow2 ** -0.5)  # == 2^{-ceil(log2 m)/2}
    term3 = 2.0 / m
    term4 = 1.0 / T
    return term1 + term2 + term3 + term4

# -------------------------- Neighbor wiring (d-regular) ------------------

def neighbors_circulant(C: int, d: int = 4) -> list[set[int]]:
    if d % 2 != 0 or d >= C:
        raise ValueError("d must be even and < C.")
    nbrs = []
    for i in range(C):
        s = set()
        for step in range(1, d // 2 + 1):
            s.add((i - step) % C)
            s.add((i + step) % C)
        nbrs.append(s)
    return nbrs

# --------------------- Holographic holonomy (boundary) -------------------

def holonomy_boundary_closure(phi_grid: np.ndarray) -> dict:
    """
    Toy holographic test on a boundary grid (SxS).
    Discrete curl per plaquette:
        curl[i,j] = dfx[i,j] + dfy[i+1,j] - dfx[i,j+1] - dfy[i,j]
    with wrapping to (-π,π].
    """
    Sx, Sy = phi_grid.shape
    dfx = angles_mod(np.diff(phi_grid, axis=0, append=phi_grid[:1, :]))
    dfy = angles_mod(np.diff(phi_grid, axis=1, append=phi_grid[:, :1]))

    # compute curl on plaquettes (Sx-1, Sy-1)
    curl = angles_mod(
        dfx[:-1, :-1] + dfy[1:, :-1] - dfx[:-1, 1:] - dfy[:-1, :-1]
    )

    plaquettes = np.abs(curl).ravel()
    med = float(np.median(plaquettes))
    mean = float(np.mean(plaquettes))
    perimeter = float(Sx + Sy)
    kappa = mean / max(perimeter, 1.0)
    return dict(median_curl=med, mean_curl=mean, kappa=kappa)


def make_boundary_grid(S: int = 16, kappa: float = 0.10, seed: int = 0) -> np.ndarray:
    """
    Build a phase boundary grid with a tunable 'vortex/flux' level kappa.
    kappa≈0 → near-conservative field (SAT-like), higher → UNSAT-like.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2*np.pi, S, endpoint=False)
    y = np.linspace(0, 2*np.pi, S, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    # base smooth pattern + curl injection
    phi = 0.2*np.sin(X) + 0.2*np.cos(Y) + kappa * np.sin(X + Y)
    # small noise for genericity
    phi += 0.03 * rng.standard_normal(size=phi.shape)
    return angles_mod(phi)

# ------------------------------- CLI tasks -------------------------------

def run_unsat(C, cR, rho, zeta0, d=4, seed=42):
    R = max(1, int(math.ceil(cR * math.log(max(2, C)))))
    T = 3 * R
    Phi, locks = schedule_unsat_hadamard(C, R, rho, zeta0, L=3, seed=seed, return_locks=True)
    G_all = gram_complex(Phi)
    mu, lam = top_mu_lambda(G_all)

    nbrs = neighbors_circulant(C, d=d)
    s2_all = edge_stats(G_all, nbrs)

    G_lock = gram_lock_only(Phi, locks)
    s2_lock = edge_stats(G_lock, nbrs)

    m = len(locks[0])
    kappa = kappa_S2(m, T, zeta0)
    print(f"UNSAT-Hadamard :: C={C}, R={R}, T={T}, m={m}, zeta0={zeta0}")
    print(f"mu={mu:.4f}, lambda_max={lam:.2f}")
    print("S2 all-T:      avg_edge={:.4f}, avg_row_sum={:.4f}".format(s2_all['avg_edge'], s2_all['avg_row_sum']))
    print("S2 lock-only:  avg_edge={:.4f}, avg_row_sum={:.4f}".format(s2_lock['avg_edge'], s2_lock['avg_row_sum']))
    print("S2 bound: d*kappa_S2 ≈ {:.4f}  (d={}, kappa_S2={:.5f})".format(d*kappa, d, kappa))

def run_sat(C, cR):
    R = max(1, int(math.ceil(cR * math.log(max(2, C)))))
    Phi = schedule_sat_envelope(C, R, L=3)
    G = gram_complex(Phi)
    mu, lam = top_mu_lambda(G)
    print(f"SAT envelope :: C={C}, R={R}, T={3*R}")
    print(f"mu={mu:.4f}, lambda_max={lam:.2f}")

def run_report(C, cR, rho, zeta0, d=4, seed=42):
    R = max(1, int(math.ceil(cR * math.log(max(2, C)))))
    T = 3 * R
    Phi, locks = schedule_unsat_hadamard(C, R, rho, zeta0, L=3, seed=seed, return_locks=True)
    G_all = gram_complex(Phi); mu, lam = top_mu_lambda(G_all)
    G_lock = gram_lock_only(Phi, locks)

    nbrs = neighbors_circulant(C, d=d)
    s2_all = edge_stats(G_all, nbrs)
    s2_lock = edge_stats(G_lock, nbrs)

    m = len(locks[0]); kapp = kappa_S2(m, T, zeta0)

    print("=== AO-FINALE report ===")
    print(f"C={C}, R={R}, T={T}, m={m}, rho={rho}, zeta0={zeta0}, d={d}")
    print(f"UNSAT-Hadamard: mu={mu:.4f}, lambda_max={lam:.2f}")
    print("S2 (all-T):     max_edge={:.4f}, avg_edge={:.4f}, max_row_sum={:.4f}, avg_row_sum={:.4f}".format(
        s2_all['max_edge'], s2_all['avg_edge'], s2_all['max_row_sum'], s2_all['avg_row_sum']))
    print("S2 (lock-only): max_edge={:.4f}, avg_edge={:.4f}, max_row_sum={:.4f}, avg_row_sum={:.4f}".format(
        s2_lock['max_edge'], s2_lock['avg_edge'], s2_lock['max_row_sum'], s2_lock['avg_row_sum']))
    print(f"S2 bound: d*kappa_S2 ≈ {d*kapp:.4f}  (kappa_S2={kapp:.5f})")

def run_holo_demo(S: int = 16, kappa: float = 0.10, seed: int = 0):
    grid = make_boundary_grid(S=S, kappa=kappa, seed=seed)
    stats = holonomy_boundary_closure(grid)
    print("=== Holographic holonomy closure (boundary) ===")
    print(f"S={S}, kappa_injected={kappa}")
    print("median |curl| = {:.4f}, mean |curl| = {:.4f}, kappa(norm) = {:.5f}".format(
        stats['median_curl'], stats['mean_curl'], stats['kappa']
    ))
    print("Interpretation: low curl → integrable closure (SAT-like). High curl → flux/vir (UNSAT-like).")

# ----------------------------------- Main --------------------------------

def main():
    ap = argparse.ArgumentParser(description="AO-FINALE spectral tester + S2 + holonomy demo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # UNSAT run
    p1 = sub.add_parser("run_unsat")
    p1.add_argument("--C", type=int, default=1000)
    p1.add_argument("--cR", type=float, default=15.0)
    p1.add_argument("--rho_lock", type=float, default=0.50)
    p1.add_argument("--zeta0", type=float, default=0.40)
    p1.add_argument("--d", type=int, default=4)
    p1.add_argument("--seed", type=int, default=42)

    # SAT run
    p2 = sub.add_parser("run_sat")
    p2.add_argument("--C", type=int, default=1000)
    p2.add_argument("--cR", type=float, default=15.0)

    # S2 report
    p3 = sub.add_parser("report_s2")
    p3.add_argument("--C", type=int, default=1000)
    p3.add_argument("--cR", type=float, default=15.0)
    p3.add_argument("--rho_lock", type=float, default=0.50)
    p3.add_argument("--zeta0", type=float, default=0.40)
    p3.add_argument("--d", type=int, default=4)
    p3.add_argument("--seed", type=int, default=42)

    # Holonomy demo (boundary-only)
    p4 = sub.add_parser("holonomy_demo")
    p4.add_argument("--S", type=int, default=16)
    p4.add_argument("--kappa", type=float, default=0.10)
    p4.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if args.cmd == "run_unsat":
        run_unsat(args.C, args.cR, args.rho_lock, args.zeta0, d=args.d, seed=args.seed)
    elif args.cmd == "run_sat":
        run_sat(args.C, args.cR)
    elif args.cmd == "report_s2":
        run_report(args.C, args.cR, args.rho_lock, args.zeta0, d=args.d, seed=args.seed)
    elif args.cmd == "holonomy_demo":
        run_holo_demo(S=args.S, kappa=args.kappa, seed=args.seed)
    else:
        raise SystemExit(1)

if __name__ == "__main__":
    main()

