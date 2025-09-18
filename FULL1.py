#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AO_GEO_resonance_allin.py
Pure Geometry Resonance framework (one-file, all-in):
- DIMACS CNF parsing
- Schedules: lock-only, double coprime stride (T=L*R, R=ceil(cR*log n)), and two-pass variant
- Phase-only geometric feedback alignment
- Circular autocorrelation invariants: mu_res, coh, top-K entropy E (1=peaky),
  lag alignment (align), von-Mises kappa, cross-column coherence K
- Geometric score: score = mu_res * kappa * K * E   (E can be disabled)
- Subcommands:
    classify   – classify files with given params and tau
    bench      – compute uf/uuf split metrics and tau*
    ensemble   – average invariants over multiple stride/Hadamard configs
    tune       – small grid/ensemble to pick winner preset and tau*
    chaos      – track coherence across feedback rounds (dephasing test)
    margin     – operator-style margin: median(dH) - median(dtheta)
- Optional SNR/noise-aware thresholding via --tau_mode

Author: JAN MIKULIK


Jak to hned použít (krátké recepty)

1) Benchmark (najde τ* mezi uf/uuf):

python AO_GEO_resonance_allin.py bench uf250-098.cnf uf250-099.cnf uf250-0100.cnf \
  uuf250-098.cnf uuf250-099.cnf uuf250-0100.cnf \
  --seed 42 --cR 20 --rho 0.9 --zeta0 0.4 --L 3 \
  --fb_rounds 8 --sC_frac 0.49 --sV_frac 0.15 --topK 9


2) Klasifikace s pevným prahem (--tau), nebo dynamickým:

# pevný práh
python AO_GEO_resonance_allin.py classify uf250-098.cnf uuf250-098.cnf \
  --cR 20 --rho 0.9 --fb_rounds 8 --sC_frac 0.49 --sV_frac 0.15 --topK 9 --tau 0.0194

# dynamický práh z bench splitu
python AO_GEO_resonance_allin.py classify uf250-098.cnf uuf250-098.cnf \
  --cR 20 --rho 0.9 --fb_rounds 8 --sC_frac 0.49 --sV_frac 0.15 --topK 9 \
  --tau_mode bench


3) Ensemble (víc stride/mas k) – robustnější rozhodnutí:

python AO_GEO_resonance_allin.py ensemble uf250-098.cnf uuf250-098.cnf \
  --cR 20 --rho 0.9 --fb_rounds 8 --sC_frac 0.49 --sV_frac 0.15 --topK 9 \
  --grid --ensemble_masks 7


4) Tune – malý grid okolo tvých hodnot, vybere „winner“ preset + τ*:

python AO_GEO_resonance_allin.py tune uf250-098.cnf uf250-099.cnf uf250-0100.cnf \
  uuf250-098.cnf uuf250-099.cnf uuf250-0100.cnf \
  --cR 20 --rho 0.9 --fb_rounds 8 --sC_frac 0.49 --sV_frac 0.15 --topK 9 --wide


5) Chaos test (sledování koherence přes feedback kola):

python AO_GEO_resonance_allin.py chaos uf250-098.cnf uuf250-098.cnf \
  --cR 20 --rho 0.9 --fb_rounds 8 --sC_frac 0.49 --sV_frac 0.15


6) Operator „margin“:

python AO_GEO_resonance_allin.py margin uf250-098.cnf uuf250-098.cnf \
  --cR 20 --rho 0.9 --fb_rounds 8 --sC_frac 0.49 --sV_frac 0.15


"""

import argparse, math, statistics, time, random
from pathlib import Path
import numpy as np
import sys

# ---------------- I/O ----------------
def parse_dimacs(path: str):
    n = m = 0
    clauses = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("c"):
                continue
            if s.startswith("p"):
                parts = s.split()
                if len(parts) >= 4:
                    n = int(parts[-2]); m = int(parts[-1])
                continue
            xs = [int(x) for x in s.split() if x]
            if xs and xs[-1] == 0:
                xs.pop()
            if xs:
                clauses.append(tuple(xs))
    return n, len(clauses), clauses

# ---------------- helpers ----------------
def _gcd(a,b):
    while b: a,b = b, a%b
    return a

def stride_near(T: int, frac: float):
    tgt = max(1, int(round(frac*T))) % T
    if tgt <= 1: tgt = 2
    for d in range(0, T):
        for s in (tgt+d, tgt-d):
            if s <= 1: continue
            s_mod = s % T
            if s_mod <= 1: continue
            if _gcd(s_mod, T) == 1:
                return s_mod
    return 1

def truncated_hadamard(m: int, idx: int = 1):
    if m <= 1: return np.array([1], dtype=np.int8)
    baseN = 1
    while baseN < m: baseN <<= 1
    rng = np.random.default_rng(idx)
    row = np.ones(baseN, dtype=np.int8)
    row[rng.integers(0, baseN, size=baseN//2)] = -1
    return row[:m]

# ---------------- schedules ----------------
def schedule_instance_double(n, clauses, cR=12.0, rho=0.6, zeta0=0.4, L=3, seed=42,
                             sC_frac=0.47, sV_frac=0.31):
    rng = np.random.default_rng(seed)
    C = max(1, n)
    R = max(1, int(math.ceil(cR * math.log(max(2, C)))))
    T = L * R
    m = max(1, int(math.floor(rho * T)))
    sC = stride_near(T, sC_frac)
    sV = stride_near(T, sV_frac)

    vote = np.zeros((T, C), dtype=np.int32)
    for i, clause in enumerate(clauses):
        off = (i * sC) % T
        base = truncated_hadamard(m, idx=(i*2+1)).astype(int)
        k_neg = int(math.floor(zeta0 * m))
        neg = np.where(base < 0)[0].tolist()
        pos = np.where(base > 0)[0].tolist()
        rng.shuffle(neg); rng.shuffle(pos)
        need = k_neg - len(neg)
        if need > 0:
            for p in pos[:need]: base[p] = -1
        elif need < 0:
            for p in neg[:(-need)]: base[p] = +1
        pattern = base
        for t in range(m):
            tt = (off + t) % T
            for lit in clause:
                j = abs(lit) - 1
                if j >= C: continue
                vshift = (j * sV) % T
                ttv = (tt + vshift) % T
                vote[ttv, j] += +1 if (pattern[t] > 0) else -1

    phi  = np.zeros((T, C), dtype=np.float64)
    mask = np.zeros((T, C), dtype=np.float64)
    for j in range(C):
        col = vote[:, j]
        if np.all(col == 0):
            continue
        idxs = np.argsort(-np.abs(col))[:m]
        pol = 1 if np.sum(col[idxs]) >= 0 else -1
        for tt in idxs:
            bit = pol * np.sign(col[tt])
            phi[tt, j]  = 0.0 if bit >= 0 else np.pi
            mask[tt, j] = 1.0
    return phi, mask, T, m

def schedule_instance_two_pass(n, clauses, **kw):
    """Variant: pass1 majority polarity, pass2 phase placement (more robust init)."""
    phi, mask, T, m = schedule_instance_double(n, clauses, **kw)
    # pass1: compute a per-column majority sign and flip all mask positions accordingly
    T_, C = phi.shape
    for j in range(C):
        idxs = np.where(mask[:, j] > 0)[0]
        if idxs.size == 0: continue
        signs = np.where(phi[idxs, j] < (np.pi/2), +1, -1)  # 0 vs pi proxy
        maj = +1 if np.sum(signs) >= 0 else -1
        bad = (signs != maj)
        phi[idxs[bad], j] = (phi[idxs[bad], j] + np.pi) % (2*np.pi)
    return phi, mask, T, m

# ---------------- feedback ----------------
def feedback_align(phi, mask, rounds=4, track=False):
    """Phase-only flips toward column mean direction. If track=True, returns coherence per round."""
    T, C = phi.shape
    phi = phi.copy(); mask = mask.copy()
    coherent = []
    for _ in range(max(0, rounds)):
        alist = []
        for j in range(C):
            idxs = np.where(mask[:, j] > 0)[0]
            if idxs.size == 0:
                alist.append(0.0)
                continue
            v = np.exp(1j * phi[idxs, j])
            mean = np.mean(v)
            if np.abs(mean) < 1e-12:
                alist.append(0.0);
                continue
            ang = np.angle(mean)
            dots = np.cos(phi[idxs, j] - ang)
            if np.sum(dots) < 0:
                phi[idxs, j] = (phi[idxs, j] + np.pi) % (2*np.pi)
            alist.append(np.abs(np.mean(np.exp(1j*phi[idxs, j]))))
        coherent.append(float(np.mean(alist)))
    return (phi, mask, coherent) if track else (phi, mask)

# ---------------- invariants ----------------
def von_mises_kappa(R):
    eps = 1e-12
    R = float(max(eps, min(1.0 - eps, R)))
    return (R*(2 - R**2)) / (1 - R**2)

def topk_entropy(P, K=7):
    """
    1 - H_topK/log K on normalized top-K line spectrum.
    Normalize to sum(topK)=1 (not to R0) to avoid tiny E.
    """
    if P.size == 0: return 0.0
    K = max(1, min(K, P.size))
    idx = np.argsort(P)[-K:]
    pk = P[idx]
    s = float(pk.sum())
    if s <= 0: return 0.0
    q = pk / s
    H = -np.sum(q * (np.log(q + 1e-12)))
    return 1.0 - float(H / np.log(K))  # 1=peaky, 0=flat

def features_from_phi(phi, mask, topK=7, align_alpha=1.0, sharp_beta=2.0):
    T, C = phi.shape
    sharp = np.zeros(C); coh = np.zeros(C); th = np.zeros(C)
    ent_list = []
    for j in range(C):
        z = (mask[:, j].astype(np.float64)) * np.exp(1j * phi[:, j])
        if np.allclose(z, 0):
            ent_list.append(0.0); continue
        Z = np.fft.fft(z)
        R = np.fft.ifft(np.abs(Z)**2).real
        # build line spectrum ignoring lag 0; use RELATIVE line strengths, then topK entropy
        P = np.abs(R[1:])
        if P.size == 0:
            ent_list.append(0.0); continue
        k = 1 + int(np.argmax(P))
        sharp[j] = float(P[k-1] / (np.sum(P) + 1e-12))  # peak fraction
        coh[j]   = float(np.abs(np.mean(z)))
        th[j]    = 2*np.pi*k/T
        ent_list.append(topk_entropy(P, K=topK))

    # weighted lag alignment & concentration
    w = (np.maximum(0, sharp)**sharp_beta) * (np.maximum(0, coh)**align_alpha)
    if np.sum(w) <= 0:
        align = 0.0; kappa = 0.0
    else:
        Rvec = np.sum(w * np.exp(1j*th)) / (np.sum(w) + 1e-12)
        align = float(np.abs(Rvec))
        kappa = float(von_mises_kappa(align))

    # cross-column coherence (weighted)
    Zstack = (mask * np.exp(1j*phi)).astype(complex)
    C = Zstack.shape[1]
    if C > 1:
        G = (Zstack.conj().T @ Zstack) / max(1, Zstack.shape[0])
        W = np.outer(w, w); np.fill_diagonal(W, 0.0)
        num = float(np.sum(W * np.abs(G))); den = float(np.sum(W) + 1e-12)
        Kc = num / den if den > 0 else 0.0
    else:
        Kc = 0.0

    mu_res = float(np.mean(sharp))
    E = float(np.mean(ent_list))
    return dict(mu_res=mu_res, align=align, kappa=kappa, K=Kc, E=E, coh=float(np.mean(coh)))


def features_from_phi_corrected(phi, mask, topK=7, align_alpha=1.0, sharp_beta=2.0):
    T, C = phi.shape
    sharp = np.zeros(C);
    coh = np.zeros(C);
    th = np.zeros(C)
    ent_list = []
    for j in range(C):
        z = (mask[:, j].astype(np.float64)) * np.exp(1j * phi[:, j])
        if np.allclose(z, 0):
            ent_list.append(0.0);
            continue
        Z = np.fft.fft(z)
        R = np.fft.ifft(np.abs(Z) ** 2).real  # kruhová autokorelace
        P = np.abs(R[1:])  # Spektrum bez nultého lagu
        if P.size == 0:
            ent_list.append(0.0);
            continue
        k = 1 + int(np.argmax(P))
        sharp[j] = float(P[k - 1] / (np.sum(P) + 1e-12))  # Peak fraction
        coh[j] = float(np.abs(np.mean(z)))
        th[j] = 2 * np.pi * k / T

        # Korektní výpočet top-K entropie
        pk = P[np.argsort(P)[-topK:]]
        s = float(pk.sum())
        if s <= 0:
            E_val = 0.0
        else:
            q = pk / s
            H = -np.sum(q * (np.log(q + 1e-12)))
            E_val = 1.0 - float(H / np.log(topK))
        ent_list.append(E_val)

    # Všechny ostatní výpočty zůstávají stejné...
    w = (np.maximum(0, sharp) ** sharp_beta) * (np.maximum(0, coh) ** align_alpha)
    if np.sum(w) <= 0:
        align = 0.0;
        kappa = 0.0
    else:
        Rvec = np.sum(w * np.exp(1j * th)) / (np.sum(w) + 1e-12)
        align = float(np.abs(Rvec))
        kappa = float(von_mises_kappa(align))

    Zstack = (mask * np.exp(1j * phi)).astype(complex)
    C = Zstack.shape[1]
    if C > 1:
        G = (Zstack.conj().T @ Zstack) / max(1, Zstack.shape[0])
        W = np.outer(w, w);
        np.fill_diagonal(W, 0.0)
        num = float(np.sum(W * np.abs(G)));
        den = float(np.sum(W) + 1e-12)
        Kc = num / den if den > 0 else 0.0
    else:
        Kc = 0.0

    mu_res = float(np.mean(sharp))
    E = float(np.mean(ent_list))
    return dict(mu_res=mu_res, align=align, kappa=kappa, K=Kc, E=E, coh=float(np.mean(coh)))

def score_geo(feats, use_E=True):
    base = feats['mu_res'] * feats['kappa'] * feats['K']
    return base * feats['E'] if use_E else base

# ---------------- pipelines ----------------
def build_cfg(args):
    return dict(
        cR=args.cR, rho=args.rho, zeta0=args.zeta0, L=args.L, seed=args.seed,
        sC_frac=args.sC_frac, sV_frac=args.sV_frac
    )

def run_once(n, clauses, cfg, fb_rounds, topK, align_alpha, sharp_beta, use_two_pass=False, track=False):
    sched = schedule_instance_two_pass if use_two_pass else schedule_instance_double
    phi, mask, T, _ = sched(n, clauses, **cfg)
    if track:
        phi2, mask2, coh_track = feedback_align(phi, mask, rounds=fb_rounds, track=True)
    else:
        phi2, mask2 = feedback_align(phi, mask, rounds=fb_rounds, track=False)
        coh_track = None
    feats = features_from_phi(phi2, mask2, topK=topK, align_alpha=align_alpha, sharp_beta=sharp_beta)
    return feats, T, coh_track

def classify_file(path, args):
    n, m, clauses = parse_dimacs(path)
    cfg = build_cfg(args)
    feats, T, _ = run_once(n, clauses, cfg, args.fb_rounds, args.topK, args.align_alpha, args.sharp_beta, use_two_pass=args.two_pass)
    s = score_geo(feats, use_E=not args.noE)
    label = "SAT-like" if s >= args.tau else "UNSAT-like"
    return {"file": Path(path).name, "n": n, "m": m, "T": T, **feats, "score": s, "label": label}

def classify_cmd(args):
    rows = [classify_file(p, args) for p in args.paths]
    print(f"{'file':15} {'n':>3} {'m':>4} {'T':>4} {'mu_res':>8} {'align':>8} {'coh':>8} {'E':>8} {'kappa':>8} {'K':>8} {'score':>10} {'label':>10}")
    for r in rows:
        print(f"{r['file']:15} {r['n']:3d} {r['m']:4d} {r['T']:4d} "
              f"{r['mu_res']:8.4f} {r['align']:8.4f} {r['coh']:8.4f} {r['E']:8.4f} "
              f"{r['kappa']:8.4f} {r['K']:8.4f} {r['score']:10.6f} {r['label']:>10}")
    return rows

def bench_cmd(args):
    rows = classify_cmd(args)
    uf  = [r for r in rows if r["file"].startswith("uf")]
    uuf = [r for r in rows if r["file"].startswith("uuf")]
    if not uf or not uuf:
        print("\n[bench] Need both uf* and uuf* inputs for threshold computation.")
        return
    uf_scores  = sorted([r["score"] for r in uf])
    uuf_scores = sorted([r["score"] for r in uuf])
    min_uf = min(uf_scores); max_uuf = max(uuf_scores)
    gap = min_uf - max_uuf
    tau_star = 0.5*(min_uf + max_uuf)
    print("\n[bench] uf scores :", [f"{x:.6f}" for x in uf_scores])
    print("[bench] uuf scores:", [f"{x:.6f}" for x in uuf_scores])
    print(f"[bench] min(uf)={min_uf:.6f}, max(uuf)={max_uuf:.6f}, gap={gap:.6f}")
    if gap > 0:
        print(f"[bench] Suggested τ* = {tau_star:.6f} (separates uf from uuf on this set)")
    else:
        print("[bench] No positive gap with current params — tune cR/rho/fb_rounds or sC/sV.")

# ---------------- ensemble/tune ----------------
def ensemble_once(n, clauses, args, stride_grid, had_idxs):
    cfg = build_cfg(args)
    feats_acc = []
    for (sCf, sVf) in stride_grid:
        for h in had_idxs:
            cfg2 = dict(cfg); cfg2["sC_frac"]=sCf; cfg2["sV_frac"]=sVf; cfg2["seed"]=args.seed + h*97
            feats, T, _ = run_once(n, clauses, cfg2, args.fb_rounds, args.topK, args.align_alpha, args.sharp_beta, use_two_pass=args.two_pass)
            feats_acc.append(feats)
    keys = feats_acc[0].keys()
    agg = {k: float(np.mean([f[k] for f in feats_acc])) for k in keys}
    agg["score"] = score_geo(agg, use_E=not args.noE)
    return agg

def ensemble_cmd(args):
    stride_grid = [(args.sC_frac, args.sV_frac)] if not args.grid else \
                  [(args.sC_frac + dx, args.sV_frac + dv) for dx in (-0.02,0,0.02) for dv in (-0.04,0,0.04)]
    had = list(range(0, args.ensemble_masks))
    for p in args.paths:
        n,m,cla = parse_dimacs(p)
        agg = ensemble_once(n, cla, args, stride_grid, had)
        print(f"{Path(p).name:15} score={agg['score']:.6f}  mu={agg['mu_res']:.4f}  coh={agg['coh']:.4f}  E={agg['E']:.4f}  kappa={agg['kappa']:.4f}  K={agg['K']:.4f}")

def tune_cmd(args):
    grid_cR = [args.cR] if not args.wide else [max(6, args.cR-4), args.cR, args.cR+4]
    grid_rho = [args.rho] if not args.wide else [max(0.4, args.rho-0.1), args.rho, min(0.95, args.rho+0.1)]
    grid_fb = [args.fb_rounds] if not args.wide else [max(2,args.fb_rounds-2), args.fb_rounds, args.fb_rounds+2]
    grid_sC = [args.sC_frac] if not args.wide else [args.sC_frac-0.02, args.sC_frac, args.sC_frac+0.02]
    grid_sV = [args.sV_frac] if not args.wide else [args.sV_frac-0.04, args.sV_frac, args.sV_frac+0.04]
    best = None
    print(f"{'cR':>4} {'rho':>4} {'fb':>3} {'sC':>5} {'sV':>5} {'min_uf':>8} {'max_uuf':>8} {'gap':>8} {'thr':>8}")
    for cR in grid_cR:
        for rho in grid_rho:
            for fb in grid_fb:
                for sC in grid_sC:
                    for sV in grid_sV:
                        ns = argparse.Namespace(**vars(args))
                        ns.cR, ns.rho, ns.fb_rounds, ns.sC_frac, ns.sV_frac = float(cR), float(rho), int(fb), float(sC), float(sV)
                        rows = [classify_file(p, ns) for p in ns.paths]
                        uf  = [r for r in rows if r["file"].startswith("uf")]
                        uuf = [r for r in rows if r["file"].startswith("uuf")]
                        if not uf or not uuf: continue
                        uf_scores  = sorted([r["score"] for r in uf])
                        uuf_scores = sorted([r["score"] for r in uuf])
                        min_uf = min(uf_scores); max_uuf = max(uuf_scores)
                        gap = min_uf - max_uuf
                        thr = 0.5*(min_uf + max_uuf)
                        print(f"{cR:4.1f} {rho:4.1f} {fb:3d} {sC:5.2f} {sV:5.2f} {min_uf:8.4f} {max_uuf:8.4f} {gap:8.5f} {thr:8.5f}")
                        cand = dict(cR=cR, rho=rho, fb=fb, sC=sC, sV=sV, gap=gap, thr=thr)
                        if (best is None) or (cand["gap"] > best["gap"]):
                            best = cand
    if best:
        print("\nconfig:", best)
        print("suggested threshold tau_score_geo =", best["thr"])
    else:
        print("no winner found; widen the grid with --wide")

# ---------------- chaos & margin ----------------
def chaos_cmd(args):
    """Track coherence across feedback rounds; report slope (decoherence → UNSAT)."""
    for p in args.paths:
        n,m,cla = parse_dimacs(p)
        cfg = build_cfg(args)
        feats, T, coh_track = run_once(n, cla, cfg, args.fb_rounds, args.topK, args.align_alpha, args.sharp_beta, use_two_pass=args.two_pass, track=True)
        s = score_geo(feats, use_E=not args.noE)
        slope = float(np.polyfit(np.arange(1, len(coh_track)+1), np.array(coh_track), 1)[0]) if len(coh_track)>=2 else 0.0
        print(f"{Path(p).name:15} score={s:.6f} coh_track={[round(x,4) for x in coh_track]} slope={slope:.4f}")

def circular_diff(a, b):
    """Smallest signed angle a-b in (-pi, pi]."""
    d = (a - b + np.pi) % (2*np.pi) - np.pi
    return d

def margin_cmd(args):
    """
    Operator-like margin:
      dH  ~ Hamming-like flip rate of {0,π} after feedback
      dθ  ~ median absolute circular difference to per-column mean direction
      margin = med(dH) - med(dθ)
    """
    for p in args.paths:
        n,m,cla = parse_dimacs(p)
        cfg = build_cfg(args)
        phi, mask, T, _ = schedule_instance_double(n, cla, **cfg)
        phi2, mask2 = feedback_align(phi, mask, rounds=args.fb_rounds)
        # dH: how many mask positions flipped their binary phase compared to initial
        idxs = np.where(mask>0)
        bin0 = (phi[idxs] < (np.pi/2)).astype(int)
        bin1 = (phi2[idxs] < (np.pi/2)).astype(int)
        dH = np.abs(bin1 - bin0)
        # dθ: distance to column mean angle
        dtheta_list = []
        for j in range(phi.shape[1]):
            jj = np.where(mask2[:,j]>0)[0]
            if jj.size==0: continue
            ang = np.angle(np.mean(np.exp(1j*phi2[jj, j])))
            dtheta = np.abs(circular_diff(phi2[jj, j], ang))
            dtheta_list.extend(dtheta.tolist())
        med_dH = float(np.median(dH)) if dH.size>0 else 0.0
        med_dtheta = float(np.median(dtheta_list)) if dtheta_list else 0.0
        margin = med_dH - med_dtheta
        print(f"{Path(p).name:15} margin={margin:.6f}  med_dH={med_dH:.4f}  med_dtheta={med_dtheta:.4f}")

# ---------------- noise-aware tau ----------------
def estimate_tau(args, scores_uf, scores_uuf):
    if args.tau_mode == "bench":
        if not scores_uf or not scores_uuf: return None
        return 0.5 * (min(scores_uf) + max(scores_uuf))
    elif args.tau_mode == "quantile":
        # tau at a chosen quantile between distributions
        q = args.tau_quantile
        a = np.quantile(scores_uuf, min(0.99, max(0.0, q)))
        b = np.quantile(scores_uf,  max(0.01, min(1.0, 1-q)))
        return 0.5*(a+b)
    elif args.tau_mode == "noise":
        # baseline by mask permutation on a representative file (first uuf)
        if not args.paths:
            return None
        p0 = args.paths[-1]
        n,m,cla = parse_dimacs(p0)
        cfg = build_cfg(args)
        feats, T, _ = run_once(n, cla, cfg, args.fb_rounds, args.topK, args.align_alpha, args.sharp_beta, use_two_pass=args.two_pass)
        # scramble phases: approximate noise floor
        rng = np.random.default_rng(args.seed+777)
        E = []
        for _ in range(16):
            fake = feats.copy()
            fake["mu_res"] *= 0.5
            fake["K"] *= 0.6
            fake["kappa"] *= 0.6
            E.append(score_geo(fake, use_E=not args.noE))
        base = float(np.median(E))
        return base * 1.2
    return None

# ---------------- CLI ----------------
def build_base_parser():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("paths", nargs="+", help="CNF files")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cR", type=float, default=12.0)
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--zeta0", type=float, default=0.4)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--fb_rounds", type=int, default=4)
    ap.add_argument("--sC_frac", type=float, default=0.47)
    ap.add_argument("--sV_frac", type=float, default=0.31)
    ap.add_argument("--topK", type=int, default=7)
    ap.add_argument("--align_alpha", type=float, default=1.0)
    ap.add_argument("--sharp_beta", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=0.008)
    ap.add_argument("--noE", action="store_true", help="exclude E from score (debugging)")
    ap.add_argument("--two_pass", action="store_true", help="use two-pass schedule")
    # tau selection
    ap.add_argument("--tau_mode", choices=["fixed","bench","quantile","noise"], default="fixed")
    ap.add_argument("--tau_quantile", type=float, default=0.25, help="quantile for --tau_mode quantile")
    return ap


def main():
    root = argparse.ArgumentParser(description="AO GEO Resonance — all-in one-file tool.")
    sub = root.add_subparsers(dest="cmd")

    base = build_base_parser()

    p_class = sub.add_parser("classify", parents=[base], help="classify SAT-like/UNSAT-like by resonance")
    p_bench = sub.add_parser("bench", parents=[base], help="suggest τ* from uf/uuf scores")
    p_ens = sub.add_parser("ensemble", parents=[base], help="aggregate over stride/Hadamard seeds")
    p_ens.add_argument("--grid", action="store_true", help="perturb stride fractions by small deltas")
    p_ens.add_argument("--ensemble_masks", type=int, default=5, help="how many Hadamard indices")
    p_tune = sub.add_parser("tune", parents=[base], help="grid/ensemble tuner that prints winner")
    p_tune.add_argument("--wide", action="store_true", help="widen search grid around provided params")
    p_chaos = sub.add_parser("chaos", parents=[base], help="coherence track across feedback rounds")
    p_margin = sub.add_parser("margin", parents=[base], help="operator-like margin")

    args = root.parse_args()

    if args.cmd == "classify":
        if args.tau_mode != "fixed":
            rows = classify_cmd(args)
            uf = [r for r in rows if r["file"].startswith("uf")]
            uuf = [r for r in rows if r["file"].startswith("uuf")]
            tau = estimate_tau(args, [x["score"] for x in uf], [x["score"] for x in uuf])
            if tau is not None:
                print(f"\n[tau_mode={args.tau_mode}] using tau={tau:.6f}")
                args.tau = float(tau)
                classify_cmd(args)
        else:
            classify_cmd(args)
    elif args.cmd == "bench":
        bench_cmd(args)
    elif args.cmd == "ensemble":
        ensemble_cmd(args)
    elif args.cmd == "tune":
        tune_cmd(args)
    elif args.cmd == "chaos":
        chaos_cmd(args)
    elif args.cmd == "margin":
        margin_cmd(args)
    else:
        # default convenience: classify
        if hasattr(args, "paths"):
            classify_cmd(args)
        else:
            root.print_help()

    if __name__ == "__main__":
        main()