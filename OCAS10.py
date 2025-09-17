#!/usr/bin/env python3
# coherence_sat.py — clean
# Instant resonance (π/2 distance, median) + multimask coherence classifier for DIMACS CNF

import argparse, math, numpy as np
from pathlib import Path

HBAR = 1.054_571_817e-34
Q_E  = 1.602_176_634e-19

# ---------- DIMACS ----------
def parse_dimacs(path):
    n = 0
    clauses = []
    cur = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('c'):
                continue
            if s.startswith('p'):
                parts = s.split()
                if len(parts) >= 4 and parts[1].lower() == 'cnf':
                    n = int(parts[2])
                cur = []
                continue
            for tok in s.split():
                lit = int(tok)
                if lit == 0:
                    if cur:
                        clauses.append(tuple(cur))
                        cur = []
                else:
                    cur.append(lit)
    if cur:
        clauses.append(tuple(cur))
    if not n and clauses:
        n = max(abs(x) for cl in clauses for x in cl)
    return n, len(clauses), clauses

# ---------- small utils ----------
def _gcd(a,b):
    while b: a,b=b,a%b
    return abs(a)

def stride_near_half(T:int)->int:
    target = max(1, T//2)
    for d in range(T):
        for cand in (target-d, target+d):
            if 1<=cand<T and _gcd(cand,T)==1:
                return cand
    return 1

def truncated_hadamard(m:int, idx:int)->np.ndarray:
    M = 1 << (max(1, m)-1).bit_length()
    row = np.empty(M, dtype=np.int8)
    mask = (M-1) & int(idx)
    for x in range(M):
        y, p = mask & x, 0
        while y:
            p ^= 1
            y &= y-1
        row[x] = -1 if p else +1
    return row[:m]

# ---------- coherence (global phasor) ----------
def mu_from_theta(theta: np.ndarray, mask: np.ndarray) -> float:
    Z = (mask * np.exp(1j*theta)).astype(np.complex128)
    C = Z.shape[1]
    m_bar = float(mask.sum(axis=0).mean())
    if m_bar <= 0:
        return 0.0
    s = np.abs(Z.sum())
    return float(s / (C * m_bar))

def build_instance_theta(n: int, clauses, cR=6.0, rho=0.5, zeta0=0.4, L=3, seed=42):
    rng = np.random.default_rng(seed)
    C = n
    R = max(1, int(math.ceil(cR*math.log(max(2, C)))))
    T = L * R
    m = max(1, int(math.floor(rho*T)))
    s_clause = stride_near_half(T)

    # ---- PASS 1: coarse vote
    vote = np.zeros((T, C), dtype=np.int32)
    for i, cl in enumerate(clauses):
        base = truncated_hadamard(m, idx=(i*2+1)).astype(int)
        k_neg = int(math.floor(zeta0*m))
        neg = np.where(base<0)[0].tolist()
        pos = np.where(base>0)[0].tolist()
        rng.shuffle(neg); rng.shuffle(pos)
        need = k_neg - len(neg)
        if need>0:
            for p in pos[:need]: base[p] = -1
        elif need<0:
            for p in neg[:(-need)]: base[p] = +1

        off = (i * s_clause) % T
        for t in range(m):
            tt = (off + t) % T
            for lit in cl:
                j = abs(lit) - 1
                if j >= C: continue
                neg_lit = (lit < 0)
                use_pi = (base[t] < 0) ^ (not neg_lit)
                vote[tt, j] += (-1 if use_pi else +1)

    assign = (vote.sum(axis=0) >= 0).astype(int)

    # ---- PASS 2: θ + mask from consistency
    theta = np.zeros((T, C), dtype=np.float64)
    mask  = np.zeros((T, C), dtype=np.float64)

    for i, cl in enumerate(clauses):
        base = truncated_hadamard(m, idx=(i*2+1)).astype(int)
        k_neg = int(math.floor(zeta0*m))
        neg = np.where(base<0)[0].tolist()
        pos = np.where(base>0)[0].tolist()
        rng.shuffle(neg); rng.shuffle(pos)
        need = k_neg - len(neg)
        if need>0:
            for p in pos[:need]: base[p] = -1
        elif need<0:
            for p in neg[:(-need)]: base[p] = +1

        off = (i * s_clause) % T
        for t in range(m):
            tt = (off + t) % T
            for lit in cl:
                j = abs(lit) - 1
                if j >= C: continue
                neg_lit = (lit < 0)
                satisfied = (assign[j] == 1 and not neg_lit) or (assign[j] == 0 and neg_lit)
                use_pi = (base[t] < 0) ^ (not satisfied)
                theta[tt, j] = 0.0 if not use_pi else np.pi
                mask[tt,  j] = 1.0

    return theta, mask, T, m

def unsat_det_mu(n, cR=6.0, rho=0.5, zeta0=0.4, L=3, seed=42):
    C = n
    R = max(1, int(math.ceil(cR*math.log(max(2, C)))))
    T = L * R
    m = max(1, int(math.floor(rho*T)))
    s = stride_near_half(T)

    phi = np.zeros((T, C)); msk = np.zeros((T, C))
    base = truncated_hadamard(m, idx=max(1, C//3)).astype(int)
    k_neg = int(math.floor(zeta0*m))
    neg = np.where(base<0)[0].tolist()
    pos = np.where(base>0)[0].tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(neg); rng.shuffle(pos)
    need = k_neg - len(neg)
    if need>0:
        for p in pos[:need]: base[p] = -1
    elif need<0:
        for p in neg[:(-need)]: base[p] = +1

    pattern = base
    if m >= 4:
        aux = truncated_hadamard(m, idx=5).astype(int)
        pattern = np.where(aux<0, -pattern, pattern)

    for j in range(C):
        off = (j*s) % T
        for t in range(m):
            tt = (off + t) % T
            phi[tt, j] = 0.0 if pattern[t]>0 else np.pi
            msk[tt, j] = 1.0

    return mu_from_theta(phi, msk)

def coherence_multimask_avg(n, clauses, K=3, S=3, **sched_kwargs):
    mus = []
    for k in range(K):
        theta, mask, T, _ = build_instance_theta(n, clauses, seed=(42 + 17*k), **sched_kwargs)
        for s in range(S):
            if s:
                theta_s = np.roll(theta, shift=(s * (T//S)), axis=0)
                mask_s  = np.roll(mask,  shift=(s * (T//S)), axis=0)
            else:
                theta_s, mask_s = theta, mask
            mus.append(mu_from_theta(theta_s, mask_s))
    mus = np.array(mus, dtype=float)
    return float(mus.min()), float(mus.mean()), float(mus.max())

# ---------- instant resonance probe ----------
def delta_phi(cnf, sigma, Phi_base, Phi_unit, x, t, omega):
    u = 0
    for cl in cnf:
        sat = False
        for lit in cl:
            j = abs(lit) - 1
            val = (sigma[j] == +1) if (lit > 0) else (sigma[j] == -1)
            if val:
                sat = True
                break
        if not sat:
            u += 1

    Phi0 = (2 * math.pi * HBAR) / Q_E
    Phi  = Phi_base + u * Phi_unit
    dphi_geo = 2.0 * math.pi * ((Phi / Phi0) % 1.0)
    total = math.pi * x + omega * t + dphi_geo
    n_star = int(round(total / (2.0 * math.pi)))
    return ((total - 2.0*math.pi*n_star + math.pi) % (2.0*math.pi)) - math.pi

def instant_resonant_guess(
    cnf,
    n,
    eps_strict=1.0e-2,
    use_dual_x=True,
    use_sigma_sym=True,
    dx=0.01,
    unit_jitter=0.02,
):
    PI, HALFPI = math.pi, math.pi/2.0

    def reduce_mod_pi(a):
        return abs(a) % PI

    def d0(a):
        x = reduce_mod_pi(a)
        return min(x, PI - x)

    def dH(a):
        x = reduce_mod_pi(a)
        return abs(x - HALFPI)

    t, omega = 1.0, 2*math.pi
    Phi0 = (2*math.pi*HBAR)/Q_E
    base_unit = Phi0/4.0

    xs_main = (0.25, 0.75) if use_dual_x else (0.25,)
    xs = []
    for xm in xs_main:
        xs.extend([xm-dx, xm, xm+dx])
    sigmas = [np.ones(n, dtype=int)]
    if use_sigma_sym:
        sigmas.append(-np.ones(n, dtype=int))
    unit_scales = [1.0, 1.0 - unit_jitter, 1.0 + unit_jitter]

    all_d0, all_dH = [], []
    strict0 = strictH = 0
    best = None
    zero_hits_x = {0.25:0, 0.75:0}

    for x in xs:
        k0 = int(round((math.pi*x + omega*t)/(2.0*math.pi)))
        frac_base = (k0 - (math.pi*x + omega*t)/(2.0*math.pi)) % 1.0
        Phi_base = frac_base*Phi0

        for sigma in sigmas:
            sgn = int(np.sign(sigma[0]) or 1)
            for scl in unit_scales:
                dphi = delta_phi(cnf, sigma, Phi_base, base_unit*scl, x, t, omega)
                dd0, ddH = d0(dphi), dH(dphi)
                all_d0.append(dd0); all_dH.append(ddH)
                if dd0 < eps_strict:
                    strict0 += 1
                    xmain = 0.25 if abs(x-0.25) <= abs(x-0.75) else 0.75
                    zero_hits_x[xmain] += 1
                if ddH < eps_strict:
                    strictH += 1
                this = (min(dd0, ddH), dphi, x, sgn, scl)
                if best is None or this[0] < best[0]:
                    best = this

    med0 = float(np.median(all_d0)) if all_d0 else float('inf')
    medH = float(np.median(all_dH)) if all_dH else float('inf')
    margin = medH - med0  # >0 => half is closer (UNSAT), <0 => zero is closer (SAT)

    if margin > 0:
        guess = "UNSAT";
        rule = "median-half-distance"
    else:
        guess = "SAT";
        rule = "median-zero-distance"

    info = {
        "rule": rule,
        "med_d0": med0,
        "med_dhalf": medH,
        "mean_d0": float(np.mean(all_d0)) if all_d0 else float('inf'),
        "mean_dhalf": float(np.mean(all_dH)) if all_dH else float('inf'),
        "strict0": int(strict0),
        "strictHalf": int(strictH),
        "zero_hits_x025": zero_hits_x.get(0.25, 0),
        "zero_hits_x075": zero_hits_x.get(0.75, 0),
        "grid_total": len(all_d0),
        "sigma_sign": best[3] if best else None,
        "margin": margin
    }
    return guess, (best[1] if best else 0.0), info


# ---------- top-level decision ----------
def classify_file(
    path,
    cR=6.0, rho=0.5, zeta0=0.4, L=3,
    K=3, S=3,
    tau_abs=0.26, ratio_rel=3.0,
    eps=0.008,
    use_dual_x=True, use_sigma_sym=True,
    res_min_margin=0.10,      # ~0.1 rad ~ 5.7°
    res_min_strict=2          # aspoň 2 striktní zásahy
):
    n, m, cls = parse_dimacs(path)

    # 1) resonance probe (named args!)
    guess_res, _, info_res = instant_resonant_guess(
        cls, n,
        eps_strict=eps,
        use_dual_x=use_dual_x,
        use_sigma_sym=use_sigma_sym,
        dx=0.01,
        unit_jitter=0.02
    )

    # 2) multimask coherence
    mu_min, mu_avg, mu_max = coherence_multimask_avg(n, cls, K=K, S=S, cR=cR, rho=rho, zeta0=zeta0, L=L)
    mu_uns = unsat_det_mu(n, cR=cR, rho=rho, zeta0=zeta0, L=L)
    tau = max(tau_abs, ratio_rel * mu_uns)
    guess_mu = "SAT" if mu_avg >= tau else "UNSAT"

    # 3) combine: resonance overrides only if strong
    margin = info_res["margin"]  # >0 => half closer (UNSAT)
    # margin > 0 => half closer (UNSAT), margin < 0 => zero closer (SAT)
    strong_res = (abs(margin) >= res_min_margin) and \
                 ((info_res["strictHalf"] >= res_min_strict) or (info_res["strict0"] >= res_min_strict))

    if strong_res:
        final = "UNSAT" if margin > 0 else "SAT"
        source = "resonance"
    else:
        final = guess_mu
        source = "coherence"

    out = {
        "file": Path(path).name,
        "n": n, "m": m,
        "guess": final, "source": source,
        "mu_min": mu_min, "mu_avg": mu_avg, "mu_max": mu_max,
        "mu_uns": mu_uns, "tau": tau,
        "res_rule": info_res["rule"],
        "med_d0": info_res["med_d0"],
        "med_dhalf": info_res["med_dhalf"],
        "margin": margin,
        "strict0": info_res["strict0"],
        "strictHalf": info_res["strictHalf"]
    }
    return out

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Resonance + multimask coherence SAT/UNSAT classifier")
    ap.add_argument("files", nargs="+", help="DIMACS CNF files")
    ap.add_argument("--cR", type=float, default=6.0)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--zeta0", type=float, default=0.4)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--S", type=int, default=3)
    ap.add_argument("--tau_abs", type=float, default=0.26)
    ap.add_argument("--ratio_rel", type=float, default=3.0)
    ap.add_argument("--eps", type=float, default=0.008)  # strict tolerance for resonance
    ap.add_argument("--no_dual_x", action="store_true")
    ap.add_argument("--no_sigma_sym", action="store_true")
    ap.add_argument("--res_min_margin", type=float, default=0.10)
    ap.add_argument("--res_min_strict", type=int,   default=2)
    args = ap.parse_args()

    use_dual_x   = not args.no_dual_x
    use_sigma_sym= not args.no_sigma_sym

    for fn in args.files:
        out = classify_file(
            fn,
            cR=args.cR, rho=args.rho, zeta0=args.zeta0, L=args.L,
            K=args.K, S=args.S,
            tau_abs=args.tau_abs, ratio_rel=args.ratio_rel,
            eps=args.eps,
            use_dual_x=use_dual_x, use_sigma_sym=use_sigma_sym,
            res_min_margin=args.res_min_margin,
            res_min_strict=args.res_min_strict
        )
        print(out)

if __name__ == "__main__":
    main()
