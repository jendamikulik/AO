#!/usr/bin/env python3
# engage_holo.py — Holographic SAT/UNSAT: Resonance + Faces + Coherence (CLI)
# Usage:
#   python engage_holo.py sample.cnf uf250-098.cnf ...

import argparse, math, json, sys
from pathlib import Path
import numpy as np

# ------- physical constants (only set the unit scale; ratios matter) -------
HBAR = 1.054_571_817e-34
Q_E  = 1.602_176_634e-19
PHI0 = (2.0*math.pi*HBAR)/Q_E

# ------------------------------- DIMACS ------------------------------------
def parse_dimacs(path):
    n, clauses, cur = 0, [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('c'): continue
            if s.startswith('p'):
                parts = s.split()
                if len(parts) >= 4 and parts[1].lower() == 'cnf':
                    n = int(parts[2])
                cur = []
                continue
            for tok in s.split():
                lit = int(tok)
                if lit == 0:
                    if cur: clauses.append(tuple(cur)); cur=[]
                else:
                    cur.append(lit)
    if cur: clauses.append(tuple(cur))
    if not n and clauses:
        n = max(abs(x) for cl in clauses for x in cl)
    return n, len(clauses), clauses

# ---------------------------- small utilities ------------------------------
def truncated_hadamard(m:int, idx:int)->np.ndarray:
    """One Walsh–Hadamard row (parity of popcount) truncated to m."""
    M = 1 << (max(1, m)-1).bit_length()
    row = np.empty(M, dtype=np.int8)
    mask = (M-1) & int(idx)
    for x in range(M):
        y, p = mask & x, 0
        while y: p ^= 1; y &= y-1
        row[x] = -1 if p else +1
    return row[:m]

def stride_near_half(T:int)->int:
    """Pick stride ~ T/2 coprime-ish with T."""
    target = max(1, T//2)
    for d in range(T):
        for cand in (target-d, target+d):
            if 1 <= cand < T and math.gcd(cand, T) == 1:
                return cand
    return 1

# -------------------------- CNF evaluation core ----------------------------
def unsat_count(cnf, sigma):
    """sigma in {+1,-1}^n, +1=True. Returns #unsatisfied clauses."""
    u = 0
    for cl in cnf:
        sat = False
        for lit in cl:
            j = abs(lit) - 1
            val = (sigma[j] == +1) if (lit > 0) else (sigma[j] == -1)
            if val: sat=True; break
        if not sat: u += 1
    return u

# ----------------------------- resonance op --------------------------------
def dmod_pi(a):  # wrap to (-π, π]
    return ((a + math.pi) % (2.0*math.pi)) - math.pi

def dist0(a):    # distance to 0 (mod π)
    y = abs(a) % math.pi
    return min(y, math.pi - y)

def distH(a):    # distance to π/2 (mod π)
    y = abs(a) % math.pi
    return abs(y - (math.pi/2.0))

def delta_phi_from_u(u, x, t, omega, Phi_base, Phi_unit):
    Phi = Phi_base + u*Phi_unit
    dphi_geo = 2.0*math.pi * ((Phi / PHI0) % 1.0)
    total = math.pi*x + omega*t + dphi_geo
    n_star = int(round(total / (2.0*math.pi)))
    return dmod_pi(total - 2.0*math.pi*n_star)

def resonance_probe(cnf, n,
                    xs_main=(0.25,0.75), dx=0.01,
                    unit_scales=(1.00, 0.98, 1.02),
                    use_sigma_sym=True):
    """
    Grid over x, small jitter, Φ_unit scales, and σ in {+1,-1}.
    Returns label, best_dphi, and detailed info including margin = med(dH) - med(d0).
    Decision (pure resonance): SAT if margin > +gamma ; UNSAT if margin < -gamma (gamma handled by arbiter).
    """
    t, omega = 1.0, 2.0*math.pi
    Phi_unit0 = PHI0/4.0

    xs = [xm+dd for xm in xs_main for dd in (-dx,0.0,+dx)]
    sigmas = [np.ones(n, dtype=int)]
    if use_sigma_sym:
        sigmas.append(-np.ones(n, dtype=int))

    d0s, dHs = [], []
    best = None  # (min(d0,dH), dphi, x, sigma_sign, scale)

    for x in xs:
        k0 = int(round((math.pi*x + omega*t)/(2.0*math.pi)))
        frac = (k0 - (math.pi*x + omega*t)/(2.0*math.pi)) % 1.0
        Phi_base = frac*PHI0

        for s in sigmas:
            sgn = int(np.sign(s[0]) or 1)
            u = unsat_count(cnf, s)
            for scl in unit_scales:
                dphi = delta_phi_from_u(u, x, t, omega, Phi_base, Phi_unit0*scl)
                a0, aH = dist0(dphi), distH(dphi)
                d0s.append(a0); dHs.append(aH)
                cand = (min(a0,aH), dphi, x, sgn, scl)
                if best is None or cand[0] < best[0]:
                    best = cand

    med0, medH = float(np.median(d0s)), float(np.median(dHs))
    margin = medH - med0   # >>> CORRECT SIGN: positive means ZERO closer (SAT)
    rule = "median-zero-distance" if margin > 0 else "median-half-distance" if margin < 0 else "median-tie"
    nominal = "SAT" if margin > 0 else "UNSAT" if margin < 0 else "UNKNOWN"

    info = dict(rule=rule, med_d0=med0, med_dhalf=medH, margin=margin,
                strict0=int(np.sum(np.array(d0s) < 1e-2)),
                strictHalf=int(np.sum(np.array(dHs) < 1e-2)))
    return nominal, (best[1] if best else 0.0), info

# --------------------------- faces (hypercube) ------------------------------
def faces_probe(cnf, n, num_faces=64, seed=123,
                xs_main=(0.25,0.75), dx=0.01,
                unit_scales=(1.00, 0.98, 1.02)):
    """
    Sample many σ masks (Hadamard/Rademacher) → aggregate resonance distances.
    More 'boundary coverage' => robustness against adversarial aliasing.
    Returns a resonance-style label for faces, plus dispersion info.
    """
    rng = np.random.default_rng(seed)

    # build a face bank
    faces = []
    # Prefer Hadamard rows if possible (structured coverage); fall back to random Rademacher.
    Hcount = min(num_faces, 256)
    for k in range(Hcount):
        row = truncated_hadamard(max(1, n), idx=(2*k+1)).astype(int)
        faces.append(row[:n])
    while len(faces) < num_faces:
        faces.append(rng.choice([-1, +1], size=n).astype(int))

    # scan as in resonance_probe, but over face bank
    t, omega = 1.0, 2.0*math.pi
    Phi_unit0 = PHI0/4.0
    xs = [xm+dd for xm in xs_main for dd in (-dx,0.0,+dx)]

    d0s, dHs = [], []
    # track how "spread out" faces are (rough entropy proxy)
    sigma_means = []

    for sigma in faces:
        sigma_means.append(abs(np.mean(sigma)))
        for x in xs:
            k0 = int(round((math.pi*x + omega*t)/(2.0*math.pi)))
            frac = (k0 - (math.pi*x + omega*t)/(2.0*math.pi)) % 1.0
            Phi_base = frac*PHI0

            u = unsat_count(cnf, sigma)
            for scl in unit_scales:
                dphi = delta_phi_from_u(u, x, t, omega, Phi_base, Phi_unit0*scl)
                d0s.append(dist0(dphi))
                dHs.append(distH(dphi))

    med0, medH = float(np.median(d0s)), float(np.median(dHs))
    margin = medH - med0   # positive => ZERO closer => SAT
    rule = "faces:median-zero" if margin > 0 else "faces:median-half" if margin < 0 else "faces:median-tie"
    nominal = "SAT" if margin > 0 else "UNSAT" if margin < 0 else "UNKNOWN"
    sigma_scatter = float(np.std(np.array(sigma_means)))  # lower -> balanced faces set

    info = dict(rule=rule, med_d0=med0, med_dhalf=medH, margin=margin,
                sigma_scatter=sigma_scatter, eps_strict=1e-2, num_faces=len(faces))
    return nominal, info

# --------------------------- coherence projector ---------------------------
def mu_from_theta(theta: np.ndarray, mask: np.ndarray) -> float:
    Z = (mask * np.exp(1j*theta)).astype(np.complex128)
    C = Z.shape[1]
    m_bar = float(mask.sum(axis=0).mean())
    if m_bar <= 0: return 0.0
    s = np.abs(Z.sum())
    return float(s / (C * m_bar))

def build_theta_mask(n, clauses, cR=6.0, rho=0.5, zeta0=0.4, L=3, seed=42):
    rng = np.random.default_rng(seed)
    C = n
    R = max(1, int(math.ceil(cR*math.log(max(2, C)))))
    T = L * R
    m = max(1, int(math.floor(rho*T)))
    stride = stride_near_half(T)

    # PASS 1: vote
    vote = np.zeros((T, C), dtype=np.int32)
    for i, cl in enumerate(clauses):
        base = truncated_hadamard(m, idx=(i*2+1)).astype(int)
        # contrast control to approx. fraction of negatives
        k_neg = int(math.floor(zeta0*m))
        neg = np.where(base<0)[0].tolist()
        pos = np.where(base>0)[0].tolist()
        rng.shuffle(neg); rng.shuffle(pos)
        need = k_neg - len(neg)
        if need>0:
            for p in pos[:need]: base[p] = -1
        elif need<0:
            for p in neg[:(-need)]: base[p] = +1

        off = (i * stride) % T
        for t in range(m):
            tt = (off + t) % T
            for lit in cl:
                j = abs(lit) - 1
                use_pi = (base[t] < 0) ^ (lit > 0)
                vote[tt, j] += (-1 if use_pi else +1)
    assign = (vote.sum(axis=0) >= 0).astype(int)  # 1=True, 0=False

    # PASS 2: theta + mask
    theta = np.zeros((T, C)); mask = np.zeros((T, C))
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

        off = (i * stride) % T
        for t in range(m):
            tt = (off + t) % T
            for lit in cl:
                j = abs(lit) - 1
                sat = (assign[j]==1 and lit>0) or (assign[j]==0 and lit<0)
                use_pi = (base[t] < 0) ^ (not sat)
                theta[tt, j] = 0.0 if not use_pi else math.pi
                mask[tt,  j] = 1.0
    return theta, mask

def coherence_mu_avg(n, clauses, K=3, S=3, **sched_kwargs):
    mus = []
    for k in range(K):
        theta, mask = build_theta_mask(n, clauses, seed=(42 + 17*k), **sched_kwargs)
        T = theta.shape[0]
        for s in range(S):
            if s:
                theta_s = np.roll(theta, shift=(s*(T//S)), axis=0)
                mask_s  = np.roll(mask,  shift=(s*(T//S)), axis=0)
            else:
                theta_s, mask_s = theta, mask
            mus.append(mu_from_theta(theta_s, mask_s))
    mus = np.array(mus, float)
    return float(mus.min()), float(mus.mean()), float(mus.max())

def unsat_mu_baseline(n, **sched_kwargs):
    # deterministic “conflict-like” reference
    C = n
    R = max(1, int(math.ceil(sched_kwargs.get('cR',6.0)*math.log(max(2, C)))))
    T = (sched_kwargs.get('L',3)) * R
    m = max(1, int(math.floor(sched_kwargs.get('rho',0.5)*T)))
    stride = stride_near_half(T)

    theta = np.zeros((T, C)); mask = np.zeros((T, C))
    base  = truncated_hadamard(m, idx=max(1, C//3)).astype(int)
    if m >= 4:
        aux = truncated_hadamard(m, idx=5).astype(int)
        base = np.where(aux<0, -base, base)
    for j in range(C):
        off = (j*stride) % T
        for t in range(m):
            tt = (off+t) % T
            theta[tt, j] = 0.0 if base[t]>0 else math.pi
            mask[tt,  j] = 1.0
    return mu_from_theta(theta, mask)

# -------------------------------- arbiter ----------------------------------
def classify_cnf(cnf, n,
                 gamma=0.10,       # dead-zone [rad] for resonance/faces
                 faces=64,         # # of σ faces
                 tau_abs=0.26, ratio_rel=3.0,
                 sched_kwargs=None):
    if sched_kwargs is None:
        sched_kwargs = dict(cR=6.0, rho=0.5, zeta0=0.4, L=3)

    # 1) resonance (global)
    r_label, dphi, r_info = resonance_probe(cnf, n)

    # 2) faces (boundary coverage) only if resonance is near dead-zone
    if -gamma < r_info["margin"] < +gamma:
        f_label, f_info = faces_probe(cnf, n, num_faces=faces)
        boundary_label, boundary_src = f_label, "faces"
        boundary_pack = ("faces_rule", f_info["rule"],
                         "faces_med_d0", f_info["med_d0"],
                         "faces_med_dhalf", f_info["med_dhalf"],
                         "faces_margin", f_info["margin"],
                         "faces_sigma", f_info["sigma_scatter"])
    else:
        boundary_label, boundary_src = r_label, "resonance"
        boundary_pack = ("res_rule", r_info["rule"],
                         "res_med_d0", r_info["med_d0"],
                         "res_med_dhalf", r_info["med_dhalf"],
                         "res_margin", r_info["margin"])

    # 3) coherence gate
    mu_min, mu_avg, mu_max = coherence_mu_avg(n, cnf, **sched_kwargs)
    mu_uns = unsat_mu_baseline(n, **sched_kwargs)
    tau = max(tau_abs, ratio_rel*mu_uns)
    mu_label = "SAT" if mu_avg >= tau else "UNSAT"

    # 4) arbiter
    #    Prefer boundary (res/faces) if far from dead-zone; otherwise let coherence decide.
    if boundary_src == "resonance" and r_info["margin"] >= +gamma:
        preliminary, source = "SAT", "resonance"      # zero closer
    elif boundary_src == "resonance" and r_info["margin"] <= -gamma:
        preliminary, source = "UNSAT", "resonance"    # half closer
    elif boundary_src == "faces" and abs(mu_avg - tau) < 0.02:  # if coherence inconclusive, trust faces
        preliminary, source = boundary_label, "faces"
    else:
        # dead-zone → coherence says
        preliminary, source = mu_label, "coherence"

    # 5) veto:
    final, src = preliminary, source
    if preliminary == "SAT" and mu_label == "UNSAT":
        final, src = "UNSAT", "coherence-veto"
    elif preliminary == "UNSAT" and mu_label == "SAT" and abs(r_info["margin"]) < (gamma+0.05):
        final, src = "SAT", "coherence-veto"

    # 6) pack
    out = {
        "guess": final,
        "source": src,
        "mu_min": mu_min, "mu_avg": mu_avg, "mu_max": mu_max,
        "mu_uns": mu_uns, "tau": tau,
        "res_rule": r_info["rule"],
        "res_med_d0": r_info["med_d0"],
        "res_med_dhalf": r_info["med_dhalf"],
        "res_margin": r_info["margin"],
    }
    # add boundary block
    for i in range(0, len(boundary_pack), 2):
        out[boundary_pack[i]] = boundary_pack[i+1]
    return out

# ---------------------------------- CLI ------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Holographic SAT/UNSAT: Resonance + Faces + Coherence")
    ap.add_argument("files", nargs="+", help="DIMACS CNF files")
    ap.add_argument("--gamma", type=float, default=0.10, help="resonance dead-zone [rad]")
    ap.add_argument("--faces", type=int, default=64, help="# sigma faces (hypercube samples)")
    ap.add_argument("--tau_abs", type=float, default=0.26)
    ap.add_argument("--ratio_rel", type=float, default=3.0)
    ap.add_argument("--cR", type=float, default=6.0)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--zeta0", type=float, default=0.4)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--json", action="store_true", help="print JSON lines")
    args = ap.parse_args()

    sched_kwargs = dict(cR=args.cR, rho=args.rho, zeta0=args.zeta0, L=args.L)

    for fn in args.files:
        n, m, cnf = parse_dimacs(fn)
        out = classify_cnf(
            cnf, n,
            gamma=args.gamma, faces=args.faces,
            tau_abs=args.tau_abs, ratio_rel=args.ratio_rel,
            sched_kwargs=sched_kwargs
        )
        row = {
            "file": Path(fn).name, "n": n, "m": m,
            **out
        }
        if args.json:
            print(json.dumps(row, ensure_ascii=False))
        else:
            print(row)

if __name__ == "__main__":
    main()
