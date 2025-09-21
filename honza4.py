#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AO_PNP_FINAL.py  —  Resonant coherence classifier (SAT-like vs UNSAT-like)
---------------------------------------------------------------------------
- Resonance-only framing (no SR, no Lorentz tricks).
- Fast Walsh row (np.bit_count if available; otherwise vectorized fallback).
- stride_near with alias_penalty + harmonic_ripple (anti-alias & anti-harmonic).
- Two-pass schedule (lock-only) with zone-mix ζ (zeta0).
- Feedback-align loop with love_gate + noise_gate_factor(+kappa, varE, time).
- Spectral witness: Gram → λ_max → μ = λ_max/C, compared to τ(α,β,safety).
- Visualizer (--viz) and directory walker (--walk).

Author: you + assistant, 2025-09-21
"""
import os, math, json, csv, argparse
from pathlib import Path
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

# --------------------- Utility ---------------------
def _next_pow2(x: int) -> int:
    n = 1
    while n < x: n <<= 1
    return n

def _gray(i: int) -> int:
    return i ^ (i >> 1)

def _gcd(a,b):
    while b: a,b = b,a%b
    return abs(a)

def _coprime(a,b) -> bool:
    return _gcd(a,b) == 1

def _bitcount_vec_uint64(arr: np.ndarray) -> np.ndarray:
    """
    Fast bitcount for uint64 vector. Prefer np.bit_count (NumPy>=1.24),
    fallback to unpackbits. Input is uint64 array.
    """
    if hasattr(np, "bit_count"):  # modern NumPy ufunc
        return np.bit_count(arr)
    # Fallback: view as bytes and use unpackbits
    # Each uint64 -> 8 bytes -> 64 bits. Sum bits per element.
    u8 = arr.view(np.uint8).reshape(arr.size, 8)
    # unpack bits per byte then sum
    bits = np.unpackbits(u8, axis=1)  # shape: (N, 8*8=64)
    return bits.sum(axis=1).astype(np.int64)

# --------------------- Walsh/Hadamard ---------------------
def _walsh_row(N: int, k: int) -> np.ndarray:
    """
    Deterministic Walsh row via Gray code and parity of bitwise intersections.
    Returns ±1 int8 vector length N.
    """
    gk = np.uint64(_gray(k))
    n = np.arange(N, dtype=np.uint64)
    bits = n & gk
    pc = _bitcount_vec_uint64(bits)  # fast bit count
    return np.where((pc & 1) == 0, 1, -1).astype(np.int8)

def truncated_hadamard(m: int, idx: int = 1) -> np.ndarray:
    if m <= 0: return np.ones(1, dtype=np.int8)
    N = _next_pow2(m)
    k = idx % N
    if k == 0: k = 1  # avoid DC
    return _walsh_row(N, k)[:m]

# --------------------- stride_near (anti-alias) ---------------------
def stride_near(T: int, frac: float, forbid=(1,2,3), search_radius=None):
    """
    Pick stride s ~ frac*T that is coprime with T and minimally aliased
    w.r.t. divisors of T and low-order harmonics. Penalize trivial strides.
    """
    if T <= 4:  # trivial
        for s in range(2, T):
            if _coprime(s, T): return s
        return max(1, T-1)

    target = int(round((frac % 1.0) * T)) % T
    target = min(max(target, 2), T-2)

    # weights (tuned)
    w_alias, w_triv, w_hr = 0.65, 2.0, 0.20
    # divisors of T (for alias checks)
    divs = [d for d in range(2, min(1024, T//2 + 1)) if T % d == 0]

    def alias_penalty(s: int) -> float:
        # penalize strides sitting near T/d multiples
        pen = 0.0
        for d in divs:
            step = T // d
            if step == 0:
                continue
            k = round(s / step)
            delta = abs(s - k*step) / max(1, step)
            if delta < 0.5:  # near multiple
                pen += (0.5 - delta)
        return pen

    def harmonic_ripple(s: int, H: int=12) -> float:
        # penalize near zeros of sin(pi*r*s/T) for small r (harmonic alias)
        acc = 0.0
        for r in range(2, H+1):
            x = math.sin(math.pi * r * s / max(1, T))
            acc += 1.0 / (1e-9 + abs(x))
        return acc

    triv = {1, 2, 3, T-1, T-2, T-3}
    golden = (math.sqrt(5)-1.0)*0.5
    prefer = int(round(golden*T)) % T

    # Candidate window (if search_radius provided)
    cand_pool = []
    if search_radius:
        lo = max(2, target - search_radius)
        hi = min(T-2, target + search_radius)
        cand_pool = [s for s in range(lo, hi+1) if _coprime(s, T)]
    else:
        cand_pool = [s for s in range(2, T-1) if _coprime(s, T)]

    if not cand_pool:  # fallback
        for s in range(2, T-1):
            if _coprime(s, T):
                cand_pool.append(s)
        if not cand_pool:
            return 1

    best_s, best_score = cand_pool[0], float('inf')
    for s in cand_pool:
        base = abs(s - target)
        pen_alias = alias_penalty(s)
        pen_triv  = w_triv if (s in triv or s in forbid) else 0.0
        pen_hr    = harmonic_ripple(s)
        reward    = 0.04 * abs(s - prefer)   # very mild sep from golden spot
        score = base + w_alias*pen_alias + pen_triv + w_hr*pen_hr + reward
        if score < best_score:
            best_s, best_score = s, score
    return best_s

def _ensure_distinct_coprime_strides(T, sC, sV):
    if not _coprime(sC, T): sC = stride_near(T, sC/T + 1e-6, search_radius=32)
    if not _coprime(sV, T): sV = stride_near(T, sV/T + 2e-6, search_radius=32)
    if sC == sV:            sV = stride_near(T, (sV+1)/T, search_radius=32)
    if not _coprime(sC, sV): sV = stride_near(T, (sV+3)/T, search_radius=48)
    return sC, sV

# --------------------- Predictors (α/β/τ) ---------------------
def predictors(eps_lock=0.01, rho=0.60, zeta0=0.30, sigma_up=0.10):
    alpha = (1.0 - eps_lock)**2
    gamma0 = rho * zeta0 - 0.5 * sigma_up
    beta  = (1.0 - gamma0)**2
    delta = 0.5 * (alpha - beta)
    return dict(alpha=alpha, beta=beta, gamma_spec=0.5*(alpha+beta),
                delta_spec=delta, gamma0=gamma0)

def auto_tau(alpha, beta, safety=None):
    if safety is None:
        safety = max(0.01, 0.25*0.5*(alpha-beta))  # conservative
    return 0.5*(alpha+beta) - safety, safety

# --------------------- Two-pass schedule with ζ ---------------------
def schedule_instance_double(n_vars, clauses, *,
    cR=15.0, rho=0.60, zeta0=0.40, L=3, seed=42,
    sC_frac=0.47, sV_frac=0.31, two_pass=True):
    """
    Build lock-only phases Φ(T×C) with two strides:
    - sC: clause-time stride
    - sV: per-variable offset stride (decorrelates columns)
    PASS1: vote per (t,j) using truncated Walsh row masked by ζ mix.
    PASS2: sort |vote| per col and pick top-m; assign phase 0 or π consistent with majority.
    """
    rng = np.random.default_rng(seed)
    C = max(1, len(clauses))
    R = max(1, int(math.ceil(cR * math.log(max(2, C)))))
    T = int(L * R)
    m = max(1, int(math.floor(rho * T)))

    # strides
    sC = stride_near(T, sC_frac, forbid=(1,2,3), search_radius=64)
    sV = stride_near(T, sV_frac, forbid=(1,2,3), search_radius=64)
    sC, sV = _ensure_distinct_coprime_strides(T, sC, sV)

    vote = np.zeros((T, C), dtype=np.int32)

    for j, clause in enumerate(clauses):
        off = (j * sC) % T
        base = truncated_hadamard(m, idx=(17*j + 11)).astype(int)

        # zone-mix ζ : proportion of -1 in lock
        k_neg = int(math.floor(zeta0 * m))
        neg_idx = np.where(base < 0)[0]
        pos_idx = np.where(base > 0)[0]
        neg_idx = neg_idx.copy(); pos_idx = pos_idx.copy()
        rng.shuffle(neg_idx); rng.shuffle(pos_idx)
        take_neg = set(neg_idx[:min(k_neg, neg_idx.size)])
        # if k_neg exceeds available negatives, fill from pos
        extra = k_neg - len(take_neg)
        if extra > 0 and pos_idx.size > 0:
            take_neg |= set(pos_idx[:min(extra, pos_idx.size)])

        slot_signs = np.array([-1 if t in take_neg else +1 for t in range(m)], dtype=int)
        # scatter across time: add per-variable stride to decorrelate pattern
        t_slots = [ (off + (t + j*sV)) % T for t in range(m) ]
        for tt, t in enumerate(t_slots):
            vote[t, j] += slot_signs[tt]

    # PASS1: per-column majority
    majority = np.sign(np.sum(vote, axis=0, dtype=np.int64))
    majority[majority == 0] = 1

    # PASS2: pick top-m slots by |vote| and assign phases (0 if aligned with majority; π otherwise)
    phi = np.ones((T, C), dtype=np.float32) * math.pi  # default outside lock = π
    mask = np.zeros((T, C), dtype=np.uint8)
    for j in range(C):
        col = vote[:, j]
        idx = np.argsort(-np.abs(col))[:m]
        for t in idx:
            mask[t, j] = 1
            # 0 if sign agrees with majority, else π (break lock lane to dilute coherence)
            phi[t, j] = 0.0 if (np.sign(col[t]) * majority[j] > 0) else math.pi

    return phi, mask, T, m, dict(sC=sC, sV=sV, R=R)

# --------------------- Gram & spectral witness ---------------------
def build_Z(phi: np.ndarray) -> np.ndarray:
    return np.exp(1j * phi)

def compute_gram(Z: np.ndarray) -> np.ndarray:
    T = Z.shape[0]
    G = (Z.conj().T @ Z) / max(1, T)
    # ensure Hermitian symmetrization (numerical)
    G = 0.5 * (G + G.conj().T)
    return G

def top_eig_pair(G: np.ndarray):
    # Efficient: power iteration for top eigenpair
    C = G.shape[0]
    v = np.ones(C, dtype=np.complex128)
    v /= npl.norm(v) + 1e-15
    lam_old = 0.0
    for _ in range(400):
        w = G @ v
        lam = float(np.vdot(v, w).real)
        nv = npl.norm(w)
        if nv < 1e-18: break
        v = w / nv
        if abs(lam - lam_old) < 1e-12: break
        lam_old = lam
    # refine λ via Rayleigh quotient
    lam = float(np.vdot(v, G @ v).real / (np.vdot(v, v).real + 1e-18))
    return lam, v

def mu_from_G(G: np.ndarray) -> float:
    lam, _ = top_eig_pair(G)
    C = G.shape[0]
    return lam / max(1, C), lam

# --------------------- Coherence, noise & gates ---------------------
def circular_kappa(col_phi: np.ndarray):
    """
    Estimate circular concentration kappa from resultant length R.
    R = |mean e^{iφ}|; small-sample corrected via standard approximation.
    """
    z = np.exp(1j * col_phi)
    R = np.abs(np.mean(z))
    if R < 1e-12:
        return 0.0, R
    # Approx. for kappa (Mardia & Jupp):
    if R < 0.53:
        kappa = 2*R + R**3 + 5*R**5/6
    elif R < 0.85:
        kappa = -0.4 + 1.39*R + 0.43/(1-R)
    else:
        kappa = 1/(R**3 - 4*R**2 + 3*R)
    return float(max(0.0, kappa)), float(R)

def noise_gate_factor(phi: np.ndarray, t_iter: int, t_max: int, *,
                      zeta0: float, base=0.9, min_fac=0.15, max_fac=1.0):
    """
    Stabilize step sizes using:
      - average kappa across columns,
      - simple energy-variance proxy,
      - time deceleration (anneal).
    Returns scalar factor in [min_fac, max_fac].
    """
    T, C = phi.shape
    # mean kappa
    ks, Rs = 0.0, 0.0
    sample_cols = min(C, 64)
    cols = np.linspace(0, C-1, sample_cols, dtype=int)
    for j in cols:
        k, R = circular_kappa(phi[:, j])
        ks += k; Rs += R
    ks /= max(1, sample_cols); Rs /= max(1, sample_cols)

    # energy variance proxy: how many π vs 0 proportion across matrix
    frac_pi = float(np.mean(np.isclose(phi, math.pi)))
    varE = frac_pi*(1-frac_pi)  # max at 0.5
    # time deceleration
    time_fac = 0.5 + 0.5*(1.0 - t_iter/max(1, t_max))  # -> 1 → 0.5

    # combine (heuristic, stable):
    fac = base * (0.8 + 0.2*np.tanh(0.25*ks)) * (1.0 - 0.3*varE) * (0.9 + 0.1*(1-zeta0)) * time_fac
    return float(min(max(min_fac, fac), max_fac))

def love_gate(phi: np.ndarray, mask: np.ndarray, *,
              strength=0.15, prefer_zero=True):
    """
    Apply 'love mask': gently bias phases inside lock towards 0 (or π).
    This is a soft, resonant 'affection' toward coherence.
    """
    if strength <= 0: return
    inside = (mask > 0)
    if not inside.any(): return
    if prefer_zero:
        phi[inside] = (1.0 - strength)*phi[inside] + strength*0.0
    else:
        phi[inside] = (1.0 - strength)*phi[inside] + strength*math.pi

def miracle_hook(phi: np.ndarray, it: int):
    """
    Placeholder for your 'miracle operators' (THEEND_UPGRADE).
    By default, it's a no-op. Plug in your private magic here.
    """
    return  # no operation by default

# --------------------- Feedback alignment (AO-resonant) ---------------------
def feedback_align(phi: np.ndarray, mask: np.ndarray, *,
                   zeta0: float, iters=20, step0=0.35, seed=42,
                   love_strength=0.12, prefer_zero=True):
    """
    AO-resonant alignment:
    - build Gram
    - take top eigenvector v
    - compute per-column complex target (weighted temporal projection)
    - update phases φ[:,j] toward arg(target) inside lock lanes
    - apply noise gate factor (kappa, varE, time), love_gate, and miracle_hook
    """
    rng = np.random.default_rng(seed)
    T, C = phi.shape
    for it in range(iters):
        Z = np.exp(1j * phi)
        G = compute_gram(Z)
        lam, v = top_eig_pair(G)         # top mode
        # per-column target angle ~ projected mean along top mode
        # We weigh rows by conjugate of v[j]: roughly align columns coherently
        target = Z @ (v.conj() / (np.abs(v)+1e-12))    # shape (T,)
        # Broadcast to columns: align each column to time-wise target
        theta = np.angle(target)                        # desired temporal phase
        # step regulation
        ng = noise_gate_factor(phi, it, iters, zeta0=zeta0)
        step = step0 * ng

        # update inside lock lines — small gradient step toward θ
        for j in range(C):
            mcol = mask[:, j].astype(bool)
            if not mcol.any():
                continue
            dphi = (theta - phi[:, j] + math.pi) % (2*math.pi) - math.pi  # wrap to [-π,π]
            phi[mcol, j] += step * dphi[mcol]

        # Love gate toward 0 (or π) inside lock
        love_gate(phi, mask, strength=love_strength*ng, prefer_zero=prefer_zero)
        # Bound phases to [0, π] outside lock to keep structure clean
        outside = (mask == 0)
        phi[outside] = np.clip(phi[outside], 0.0, math.pi)
        # Hook for your special AO miracle ops (optional)
        miracle_hook(phi, it)

    return phi

# --------------------- DIMACS CNF ---------------------
def parse_dimacs(path: str):
    s = Path(path).read_text(encoding='utf-8', errors='ignore').splitlines()
    clauses = []
    n_vars = 0
    cur = []
    for line in s:
        line = line.strip()
        if not line or line.startswith('c'):
            continue
        if line.startswith('p'):
            pr = line.split()
            if len(pr) >= 4 and pr[1].lower() == 'cnf':
                n_vars = int(pr[2])
            continue
        for tok in line.split():
            v = int(tok)
            if v == 0:
                if cur:
                    clauses.append(tuple(cur))
                    cur = []
            else:
                cur.append(v)
    if cur:
        clauses.append(tuple(cur))
    if n_vars == 0:
        mx = 0
        for cl in clauses:
            for lit in cl:
                mx = max(mx, abs(lit))
        n_vars = mx
    return n_vars, clauses

# --------------------- Classify CNF ---------------------
def classify_cnf(path: str, *,
                 cR=15.0, rho=0.50, zeta0=0.40, L=3,
                 sC_frac=0.47, sV_frac=0.31,
                 tau="auto", eps_lock=0.01, sigma_up=0.045,
                 fa_iters=24, seed=42, viz=False):
    n_vars, clauses = parse_dimacs(path)
    C = len(clauses)
    phi, mask, T, m, aux = schedule_instance_double(
        n_vars, clauses, cR=cR, rho=rho, zeta0=zeta0, L=L, seed=seed,
        sC_frac=sC_frac, sV_frac=sV_frac, two_pass=True
    )

    # predictors & tau
    bands = predictors(eps_lock=eps_lock, rho=rho, zeta0=zeta0, sigma_up=sigma_up)
    if tau == "auto":
        tau_val, safety = auto_tau(bands["alpha"], bands["beta"])
    else:
        tau_val = float(tau)

    # feedback-align refinement (resonant)
    phi = feedback_align(phi, mask, zeta0=zeta0, iters=fa_iters, seed=seed)

    # spectral witness
    Z = build_Z(phi)
    G = compute_gram(Z)
    mu, lam = mu_from_G(G)
    decision = "SAT" if mu >= tau_val else "UNSAT"

    res = {
        "file": str(path),
        "n_vars": n_vars,
        "n_clauses": C,
        "R": aux["R"], "T": T, "m": m,
        "sC": aux["sC"], "sV": aux["sV"],
        "alpha": bands["alpha"], "beta": bands["beta"],
        "tau": tau_val, "mu": mu, "lambda_max": lam,
        "decision": decision
    }

    if viz:
        fig, axs = plt.subplots(1, 2, figsize=(10,4))
        axs[0].imshow(phi.T, aspect='auto', cmap='twilight', interpolation='nearest')
        axs[0].set_title("Phases (Φ)  [rows: time, cols: clause]")
        axs[0].set_xlabel("time"); axs[0].set_ylabel("clause")
        # phasor snapshot
        t0 = np.argmax(mask.sum(axis=1))  # most-locked time slice
        zslice = np.exp(1j*phi[t0, :])
        axs[1].quiver(np.zeros(C), np.zeros(C),
                      np.cos(np.angle(zslice)), np.sin(np.angle(zslice)),
                      angles='xy', scale_units='xy', scale=1.0, width=0.002)
        axs[1].set_aspect('equal', 'box')
        axs[1].set_xlim(-1.1, 1.1); axs[1].set_ylim(-1.1, 1.1)
        axs[1].set_title(f"Phasors @t={t0}, μ={mu:.3f}, τ={tau_val:.3f}")
        plt.tight_layout()
        plt.show()

    return res

# --------------------- Walker (batch) ---------------------
def walk_dir(root: str, out_csv: str, **kwargs):
    paths = []
    for p in Path(root).rglob("*.cnf"):
        paths.append(p)
    paths.sort()
    os.makedirs(Path(out_csv).parent, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "file","n_vars","n_clauses","R","T","m","sC","sV",
            "alpha","beta","tau","mu","lambda_max","decision"
        ])
        w.writeheader()
        for p in paths:
            try:
                res = classify_cnf(str(p), **kwargs, viz=False)
                w.writerow({k:res.get(k) for k in w.fieldnames})
                # also dump per-instance JSON next to CSV
                jpath = Path(out_csv).with_suffix("")  # folder stem
                jdir = str(jpath)+"_json"
                Path(jdir).mkdir(parents=True, exist_ok=True)
                with open(os.path.join(jdir, Path(p).name + ".json"), "w", encoding="utf-8") as jf:
                    json.dump(res, jf, indent=2)
                print(f"[OK] {p} → {res['decision']}, μ={res['mu']:.4f}, τ={res['tau']:.4f}")
            except Exception as e:
                print(f"[ERR] {p}: {e}")

# --------------------- CLI ---------------------
def main():
    ap = argparse.ArgumentParser(description="AO Resonant Coherence Classifier (PNP-style)")
    ap.add_argument("--cnf", type=str, default="", help="Path to DIMACS CNF")
    ap.add_argument("--walk", type=str, default="", help="Directory to scan recursively for *.cnf")
    ap.add_argument("--out", type=str, default="results.csv", help="CSV path for --walk mode")
    ap.add_argument("--tau", default="auto", help="'auto' or explicit float threshold")
    ap.add_argument("--cR", type=float, default=15.0)
    ap.add_argument("--rho", type=float, default=0.50)
    ap.add_argument("--zeta0", type=float, default=0.40)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--sC_frac", type=float, default=0.47)
    ap.add_argument("--sV_frac", type=float, default=0.31)
    ap.add_argument("--eps_lock", type=float, default=0.01)
    ap.add_argument("--sigma_up", type=float, default=0.045)
    ap.add_argument("--fa_iters", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--viz", action="store_true")
    args = ap.parse_args()

    if args.walk:
        walk_dir(args.walk, args.out,
                 cR=args.cR, rho=args.rho, zeta0=args.zeta0, L=args.L,
                 sC_frac=args.sC_frac, sV_frac=args.sV_frac,
                 tau=args.tau, eps_lock=args.eps_lock, sigma_up=args.sigma_up,
                 fa_iters=args.fa_iters, seed=args.seed)
    elif args.cnf:
        res = classify_cnf(args.cnf, cR=args.cR, rho=args.rho, zeta0=args.zeta0, L=args.L,
                           sC_frac=args.sC_frac, sV_frac=args.sV_frac,
                           tau=args.tau, eps_lock=args.eps_lock, sigma_up=args.sigma_up,
                           fa_iters=args.fa_iters, seed=args.seed, viz=args.viz)
        print(json.dumps(res, indent=2))
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
