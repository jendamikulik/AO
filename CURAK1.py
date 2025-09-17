#!/usr/bin/env python3
# ao_holo_sat.py
# Deterministic SAT/UNSAT classifier:
#   1) instant π-rational resonance probe (median distances)
#   2) holographic hypercube-face holonomy probe (closure on Q_n faces)
#   3) optional coherence (multimask) fallback for weak/tied cases
#
# deps: numpy
# usage:
#   python ao_holo_sat.py <files...>
#   (CNF in DIMACS)

import argparse, math, numpy as np
from pathlib import Path

# ---------- physical constants for the π-rational drive ----------
HBAR = 1.054_571_817e-34
Q_E  = 1.602_176_634e-19

# ================================================================
# DIMACS
# ================================================================
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

# ================================================================
# small helpers
# ================================================================
def _wrap_pi(a):
    """wrap to (-π, π]"""
    x = (a + math.pi) % (2.0*math.pi) - math.pi
    return x

def _d0_mod(a):
    """distance to 0 on a circle with period π"""
    PI = math.pi
    x = abs(a) % PI
    return min(x, PI - x)

def _dH_mod(a):
    """distance to π/2 on a circle with period π"""
    PI2 = math.pi/2.0
    x = abs(a) % math.pi
    return abs(x - PI2)

def _rng(seed):
    return np.random.default_rng(seed)

# ================================================================
# resonance phase model
# ================================================================
def unsat_count(clauses, sigma):
    """u = #unsatisfied clauses for assignment sigma (sigma in {+1,-1}^n)"""
    u = 0
    for cl in clauses:
        sat = False
        for lit in cl:
            j = abs(lit) - 1
            val = (sigma[j] == +1) if (lit > 0) else (sigma[j] == -1)
            if val:
                sat = True
                break
        if not sat:
            u += 1
    return u

def phi_value(clauses, sigma, Phi_base, Phi_unit, x, t, omega):
    """total wrapped phase for an assignment sigma"""
    u = unsat_count(clauses, sigma)
    Phi0 = (2*math.pi*HBAR)/Q_E
    Phi  = Phi_base + u*Phi_unit
    dphi_geo = 2.0*math.pi * ((Phi / Phi0) % 1.0)
    total = math.pi*x + omega*t + dphi_geo
    return _wrap_pi(total)

def delta_phi(clauses, sigma, Phi_base, Phi_unit, x, t, omega):
    """for compatibility with previous notation (same as phi_value)"""
    return phi_value(clauses, sigma, Phi_base, Phi_unit, x, t, omega)

# ================================================================
# (1) Resonance probe (π/2 quantization via median distances)
# ================================================================
def resonance_probe(clauses, n,
                    dx=0.01,
                    unit_jitter=0.02,
                    use_dual_x=True,
                    use_sigma_sym=True,
                    eps_strict=1e-2):
    """
    Returns:
      label in {"SAT","UNSAT","UNKNOWN"},
      best_raw_delta (float),
      info (dict)
    """
    t, omega = 1.0, 2.0*math.pi
    Phi0      = (2.0*math.pi*HBAR)/Q_E
    base_unit = Phi0/4.0

    xs_main = (0.25, 0.75) if use_dual_x else (0.25,)
    xs = []
    for xm in xs_main:
        xs.extend([xm - dx, xm, xm + dx])

    sigmas = [np.ones(n, dtype=int)]
    if use_sigma_sym:
        sigmas.append(-np.ones(n, dtype=int))
    unit_scales = [1.0, 1.0 - unit_jitter, 1.0 + unit_jitter]

    all_d0, all_dH = [], []
    strict0 = 0
    strictH = 0
    best = None  # (min(d0,dH), dphi, x, sign, scale)

    for x in xs:
        k0 = int(round((math.pi*x + omega*t) / (2.0*math.pi)))
        frac_base = (k0 - (math.pi*x + omega*t)/(2.0*math.pi)) % 1.0
        Phi_base = frac_base * Phi0

        for sigma in sigmas:
            sgn = int(np.sign(sigma[0]) or 1)
            for scl in unit_scales:
                dphi = delta_phi(clauses, sigma, Phi_base, base_unit*scl, x, t, omega)
                d0 = _d0_mod(dphi)
                dH = _dH_mod(dphi)
                all_d0.append(d0); all_dH.append(dH)
                if d0 < eps_strict: strict0 += 1
                if dH < eps_strict: strictH += 1
                cand = (min(d0, dH), dphi, x, sgn, scl)
                if best is None or cand[0] < best[0]:
                    best = cand

    med0 = float(np.median(all_d0)) if all_d0 else float("inf")
    medH = float(np.median(all_dH)) if all_dH else float("inf")
    margin = medH - med0  # <0 ⇒ zero closer ⇒ SAT ; >0 ⇒ half closer ⇒ UNSAT

    if margin > 0:
        nominal = "SAT"; rule = "median-zero-distance"
    elif margin < 0:
        nominal = "UNSAT"; rule = "median-half-distance"
    else:
        nominal = "UNKNOWN"; rule = "median-tie"

    info = {
        "rule": rule,
        "med_d0": med0,
        "med_dhalf": medH,
        "margin": margin,
        "strict0": strict0,
        "strictHalf": strictH,
        "x_used": best[2] if best else None,
        "sigma_sign": best[3] if best else None,
    }
    return nominal, (best[1] if best else 0.0), info

# ================================================================
# (2) Hypercube holographic probe (faces holonomy closure)
# ================================================================
def hypercube_faces_probe(clauses, n,
                          num_faces=64,
                          seed=123,
                          dx=0.01,
                          unit_jitter=0.02,
                          eps_strict=1e-2):
    """
    Randomly sample 2D faces (i,j). For each face evaluate oriented holonomy:
      H = φ(x) - φ(x⊕e_i) - φ(x⊕e_j) + φ(x⊕e_i⊕e_j)  (wrapped to (-π,π])
    Repeat across x-jitter and unit_jitter → collect distances to {0, π/2}.
    Decide SAT if median d0 < median dH AND stability (std) is below Δσ.
    """
    rng = _rng(seed)
    t, omega = 1.0, 2.0*math.pi
    Phi0      = (2.0*math.pi*HBAR)/Q_E
    base_unit = Phi0/4.0

    xs = [0.25 - dx, 0.25, 0.25 + dx, 0.75 - dx, 0.75, 0.75 + dx]
    unit_scales = [1.0, 1.0 - unit_jitter, 1.0 + unit_jitter]

    # choose faces
    pairs = []
    all_idx = np.arange(n)
    for _ in range(num_faces):
        i, j = tuple(rng.choice(all_idx, size=2, replace=False))
        if i > j: i, j = j, i
        pairs.append((i, j))

    d0s, dHs = [], []
    # local stability (scatter) for Δσ:
    samples_for_sigma = []

    # start from two base vertices to avoid bias (all +1, all -1)
    base_vertices = [np.ones(n, dtype=int), -np.ones(n, dtype=int)]

    for (i, j) in pairs:
        for base in base_vertices:
            for x in xs:
                k0 = int(round((math.pi*x + omega*t)/(2.0*math.pi)))
                frac_base = (k0 - (math.pi*x + omega*t)/(2.0*math.pi)) % 1.0
                Phi_base = frac_base * Phi0

                for scl in unit_scales:
                    v00 = base.copy()
                    v10 = base.copy(); v10[i] *= -1
                    v01 = base.copy(); v01[j] *= -1
                    v11 = base.copy(); v11[i] *= -1; v11[j] *= -1

                    phi00 = phi_value(clauses, v00, Phi_base, base_unit*scl, x, t, omega)
                    phi10 = phi_value(clauses, v10, Phi_base, base_unit*scl, x, t, omega)
                    phi01 = phi_value(clauses, v01, Phi_base, base_unit*scl, x, t, omega)
                    phi11 = phi_value(clauses, v11, Phi_base, base_unit*scl, x, t, omega)

                    hol = _wrap_pi(phi00 - phi10 - phi01 + phi11)

                    d0s.append(_d0_mod(hol))
                    dHs.append(_dH_mod(hol))
                    samples_for_sigma.append(hol)

    d0s = np.asarray(d0s, dtype=float)
    dHs = np.asarray(dHs, dtype=float)
    med0 = float(np.median(d0s)) if d0s.size else float('inf')
    medH = float(np.median(dHs)) if dHs.size else float('inf')
    margin = medH - med0  # <0 ⇒ zero closer ⇒ SAT ; >0 ⇒ half closer ⇒ UNSAT
    sig = float(np.std(samples_for_sigma)) if samples_for_sigma else float('inf')

    # decision: prioritize geometry (closure) then σ-stability gate
    if margin > 0:
        label, rule = "SAT", "faces:median-zero"
    elif margin < 0:
        label, rule = "UNSAT", "faces:median-half"
    else:
        label, rule = "UNKNOWN", "faces:tied"

    info = {
        "rule": rule,
        "med_d0": med0,
        "med_dhalf": medH,
        "margin": margin,
        "sigma_scatter": sig,
        "eps_strict": eps_strict,
        "num_faces": int(num_faces)
    }
    return label, info

# ================================================================
# (3) Coherence (multimask) — simple, fast baseline fallback
# ================================================================
def _gcd(a,b):
    while b: a,b=b,a%b
    return abs(a)

def _stride_near_half(T:int)->int:
    target = max(1, T//2)
    for d in range(T):
        for cand in (target-d, target+d):
            if 1<=cand<T and _gcd(cand,T)==1:
                return cand
    return 1

def _truncated_hadamard(m:int, idx:int)->np.ndarray:
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

def _mu_from_theta(theta: np.ndarray, mask: np.ndarray) -> float:
    Z = (mask * np.exp(1j*theta)).astype(np.complex128)
    C = Z.shape[1]
    m_bar = float(mask.sum(axis=0).mean())
    if m_bar <= 0: return 0.0
    s = np.abs(Z.sum())
    return float(s / (C * m_bar))

def _build_instance_theta(n, clauses, cR=6.0, rho=0.5, zeta0=0.4, L=3, seed=42):
    rng = _rng(seed)
    C = n
    R = max(1, int(math.ceil(cR*math.log(max(2, C)))))
    T = L * R
    m = max(1, int(math.floor(rho*T)))
    s_clause = _stride_near_half(T)

    vote = np.zeros((T, C), dtype=np.int32)
    for i, cl in enumerate(clauses):
        base = _truncated_hadamard(m, idx=(i*2+1)).astype(int)
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

    theta = np.zeros((T, C), dtype=np.float64)
    mask  = np.zeros((T, C), dtype=np.float64)
    for i, cl in enumerate(clauses):
        base = _truncated_hadamard(m, idx=(i*2+1)).astype(int)
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

def _unsat_det_mu(n, cR=6.0, rho=0.5, zeta0=0.4, L=3, seed=42):
    C = n
    R = max(1, int(math.ceil(cR*math.log(max(2, C)))))
    T = L * R
    m = max(1, int(math.floor(rho*T)))
    s = _stride_near_half(T)

    phi = np.zeros((T, C)); msk = np.zeros((T, C))
    base = _truncated_hadamard(m, idx=max(1, C//3)).astype(int)
    k_neg = int(math.floor(zeta0*m))
    neg = np.where(base<0)[0].tolist()
    pos = np.where(base>0)[0].tolist()
    rng = _rng(seed)
    rng.shuffle(neg); rng.shuffle(pos)
    need = k_neg - len(neg)
    if need>0:
        for p in pos[:need]: base[p] = -1
    elif need<0:
        for p in neg[:(-need)]: base[p] = +1

    pattern = base
    if m >= 4:
        aux = _truncated_hadamard(m, idx=5).astype(int)
        pattern = np.where(aux<0, -pattern, pattern)
    for j in range(C):
        off = (j*s) % T
        for t in range(m):
            tt = (off + t) % T
            phi[tt, j] = 0.0 if pattern[t]>0 else np.pi
            msk[tt, j] = 1.0
    return _mu_from_theta(phi, msk)

def coherence_fallback(n, clauses,
                       cR=6.0, rho=0.5, zeta0=0.4, L=3, K=3, S=3,
                       tau_abs=0.26, ratio_rel=3.0):
    mus = []
    for k in range(K):
        theta, mask, T, _ = _build_instance_theta(n, clauses, seed=(42 + 17*k),
                                                  cR=cR, rho=rho, zeta0=zeta0, L=L)
        for s in range(S):
            if s:
                theta_s = np.roll(theta, shift=(s * (T//S)), axis=0)
                mask_s  = np.roll(mask,  shift=(s * (T//S)), axis=0)
            else:
                theta_s, mask_s = theta, mask
            mus.append(_mu_from_theta(theta_s, mask_s))
    mu_min, mu_avg, mu_max = float(np.min(mus)), float(np.mean(mus)), float(np.max(mus))
    mu_uns = _unsat_det_mu(n, cR=cR, rho=rho, zeta0=zeta0, L=L)
    tau = max(tau_abs, ratio_rel * mu_uns)
    label = "SAT" if mu_avg >= tau else "UNSAT"
    return label, {"mu_min":mu_min,"mu_avg":mu_avg,"mu_max":mu_max,"mu_uns":mu_uns,"tau":tau}

# ================================================================
# Top-level classification
# ================================================================
def classify_file(path,
                  faces=64,
                  gamma=0.10,
                  # coherence params:
                  cR=6.0, rho=0.5, zeta0=0.4, L=3, K=3, S=3,
                  tau_abs=0.26, ratio_rel=3.0):
    n, m, clauses = parse_dimacs(path)

    # 1) resonance
    res_label, dphi_best, res_info = resonance_probe(clauses, n)

    # 2) hypercube faces (holographic closure)
    face_label, face_info = hypercube_faces_probe(clauses, n, num_faces=faces)

    # 3) arbiter
    # strong resonance outside dead-zone → take it
    margin = res_info["margin"]
    if margin <= -gamma:
        preliminary, source = "SAT", "resonance"
    elif margin >= +gamma:
        preliminary, source = "UNSAT", "resonance"
    else:
        # if resonance weak, trust hypercube geometry first
        if face_label != "UNKNOWN":
            preliminary, source = face_label, "faces"
        else:
            preliminary, source = "UNKNOWN", "faces-weak"

    # 4) fallback with coherence only if still UNKNOWN or conflict
    final = preliminary
    coh_meta = {}
    if final == "UNKNOWN":
        coh_label, coh_meta = coherence_fallback(n, clauses,
                            cR=cR, rho=rho, zeta0=zeta0, L=L, K=K, S=S,
                            tau_abs=tau_abs, ratio_rel=ratio_rel)
        final, source = coh_label, "coherence"

    out = {
        "file": Path(path).name,
        "n": n, "m": m,
        "guess": final, "source": source,
        # resonance diagnostics
        "res_rule": res_info["rule"],
        "res_med_d0": res_info["med_d0"],
        "res_med_dhalf": res_info["med_dhalf"],
        "res_margin": res_info["margin"],
        # faces diagnostics
        "faces_rule": face_info["rule"],
        "faces_med_d0": face_info["med_d0"],
        "faces_med_dhalf": face_info["med_dhalf"],
        "faces_margin": face_info["margin"],
        "faces_sigma": face_info["sigma_scatter"],
    }
    out.update(coh_meta)  # add mu_* if computed
    return out

# ================================================================
# CLI
# ================================================================
def main():
    ap = argparse.ArgumentParser(description="AO-Holonomy Resonance SAT/UNSAT (resonance + hypercube faces + coherence fallback)")
    ap.add_argument("files", nargs="+", help="DIMACS CNF files")
    ap.add_argument("--faces", type=int, default=64, help="sampled hypercube faces")
    ap.add_argument("--gamma", type=float, default=0.10, help="resonance dead-zone [rad]")
    # coherence params (fallback):
    ap.add_argument("--cR", type=float, default=6.0)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--zeta0", type=float, default=0.4)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--S", type=int, default=3)
    ap.add_argument("--tau_abs", type=float, default=0.26)
    ap.add_argument("--ratio_rel", type=float, default=3.0)
    args = ap.parse_args()

    for fn in args.files:
        out = classify_file(
            fn, faces=args.faces, gamma=args.gamma,
            cR=args.cR, rho=args.rho, zeta0=args.zeta0, L=args.L,
            K=args.K, S=args.S, tau_abs=args.tau_abs, ratio_rel=args.ratio_rel
        )
        print(out)

if __name__ == "__main__":
    main()
