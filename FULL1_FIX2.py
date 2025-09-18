#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AO_GEO_resonance_full.py
Pure Geometry Resonance — kompletní verze s plnou logikou:
- DIMACS parser
- Schedules:
    * lock-only double coprime stride (T = L * R, R = ceil(cR * log max(2,n)))
    * two-pass varianta (globální polarita -> jemné rozmístění)
- Walsh–Hadamard (Sylvester) generátor v sequency pořadí + bezpečný ořez
- Geometrický feedback s annealingem, flip-gate, tie-breaker jitterem a noise gatingem
- Invarianta z kruhové autokorelace: mu_res, coh, top-K entropie E (1=peaky),
  lag-alignment align, von-Mises kappa, cross-column K
- Skóre: score = mu_res * kappa * K * E  (E lze vypnout)
- Subcommands: classify, bench, tune, ensemble, chaos, margin
"""

import argparse, math, time, statistics
from pathlib import Path
import numpy as np
import sys
# -------------------------- I/O --------------------------
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
            if xs and xs[-1] == 0: xs.pop()
            if xs: clauses.append(tuple(xs))
    return n, len(clauses), clauses

# ------------------------ helpers ------------------------
def _gcd(a,b):
    while b: a,b = b, a%b
    return abs(a)

def _coprime(a,b): return _gcd(a,b) == 1

def stride_near(T: int, frac: float, forbid=(1,2), search_radius=None):
    """
    Najdi krok s ≡ round(frac*T) (mod T), který je:
      - koprimární s T
      - není triviální (1, 2, T-1 apod.)
      - co nejblíž cíli (|s - target|), s malou penalizací aliasingu
    Fallback: nejmenší coprime > 2.
    """
    if T <= 4:
        return 1 if T <= 1 else 2 % T

    # cílová pozice
    target = int(round((frac % 1.0) * T)) % T
    if target < 2: target = 2
    if target >= T-1: target = T-2

    R = search_radius or max(8, int(0.15*T))
    best = None
    def score(s):
        # penalizuj triviální kroky a kroky blízké násobkům T/L
        triv = {1,2,T-1,T-2}
        pen = 3.0 if s in triv else 0.0
        # vyhni se krokům s malým periodickým cyklem (aliasing): dělitele T
        # „měkká“ penalizace: suma 1/d pro d | T a s ≈ k*(T/d)
        alias = 0.0
        for d in range(2, min(32, T//2)+1):
            if T % d == 0:
                k = round(s / (T/d))
                alias += 0.5 / d * (1.0/(1.0 + abs(s - k*(T//d))))
        return abs(s - target) + pen + alias

    # primárně hledej v okolí cíle
    for delta in range(0, R+1):
        for cand in (target+delta, target-delta):
            s = cand % T
            if s < 2 or s >= T-1: continue
            if not _coprime(s, T): continue
            sc = score(s)
            if best is None or sc < best[0]:
                best = (sc, s)
        if best is not None and delta > 8:
            break
    if best: return best[1]

    # fallback: vezmi nejmenší coprime > 2
    for s in range(3, T-1):
        if _coprime(s, T): return s
    return 2

def _walsh_sequency_order(N: int):
    """
    Vrátí pořadí řádků Walsh–Hadamard (sequency = počet změn znaménka).
    Pro N=2^k: sequency r je Gray(kod) r -> pořadí.
    """
    # Gray code trick: pořadí řádků dle počtu bitových přechodů
    # Prakticky stačí standardní pořadí 0..N-1 — stejně si volíme řádek indexem.
    return np.arange(N, dtype=int)

def truncated_hadamard(m: int, idx: int = 0):
    """
    Vytvoř délku m s Walsh–Hadamard strukturou:
      - zkonstruuj Sylvester H_N pro N=2^k >= m (bez explicitní NxN matice)
      - vyber řádek r = (idx mod N) v *sequency* pořadí
      - vrať prvních m hodnot z řádku (±1)
    Implementace řádku přes fast Walsh-Hadamard transform „na přání“:
      H_r[c] = (-1)^{<r, c>} (skalární součin bitů r & c).
    """
    if m <= 1:
        return np.array([1], dtype=np.int8)
    N = 1
    while N < m: N <<= 1
    r = (idx % N)
    # vygeneruj řádek bez budování celé matice: H[r, c] = (-1)^{popcount(r & c)}
    c = np.arange(m, dtype=np.uint64)
    x = (r & c).astype(np.uint64)
    # popcount
    # využij vektorové triky: python 3.10+ má int.bit_count; pro vectorizaci:
    def bitcount(u):  # u je ndarray uint64
        v = u.copy()
        v = v - ((v >> 1) & np.uint64(0x5555555555555555))
        v = (v & np.uint64(0x3333333333333333)) + ((v >> 2) & np.uint64(0x3333333333333333))
        v = (v + (v >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
        v = v + (v >> 8)
        v = v + (v >> 16)
        v = v + (v >> 32)
        return (v & np.uint64(0x7F)).astype(np.uint8)
    bc = bitcount(x)
    row = np.where((bc & 1) == 0, 1, -1).astype(np.int8)
    return row

# --------------------- schedules ---------------------
def _ensure_distinct_coprime_strides(T, sC, sV):
    # Zajisti sC != sV a coprime s T; pokud ne, posuň sV/sC k nejbližšímu coprime.
    if not _coprime(sC, T): sC = stride_near(T, sC/T + 1e-9)
    if not _coprime(sV, T): sV = stride_near(T, sV/T + 2e-9)
    if sC == sV:
        sV = stride_near(T, (sV+1)/T)
    # preferuj vzájemně coprime (není nutné, ale pomáhá)
    if not _coprime(sC, sV):
        sV = stride_near(T, (sV+3)/T)
    return sC, sV

def schedule_instance_double(n, clauses, cR=12.0, rho=0.6, zeta0=0.4, L=3, seed=42,
                             sC_frac=0.47, sV_frac=0.31, two_pass=False):
    """
    Lock-only rozvrh s dvojicí koprimárních stride:
      - time stride pro klauzule (sC)
      - per-variable shift stride (sV)
    two_pass:
      PAS 1: spočti pro každý sloupec „major pol“ z hrubých hlasů (vote)
      PAS 2: vyber top-m časy ve sloupci v souladu s major pol, a nastav fáze {0,π}
    """
    rng = np.random.default_rng(seed)
    C = max(1, n)
    R = max(1, int(math.ceil(cR * math.log(max(2, C)))))
    T = int(L * R)
    m = max(1, int(math.floor(rho * T)))

    sC = stride_near(T, sC_frac)
    sV = stride_near(T, sV_frac)
    sC, sV = _ensure_distinct_coprime_strides(T, sC, sV)

    # hlasovací matice
    vote = np.zeros((T, C), dtype=np.int32)
    for i, clause in enumerate(clauses):
        off = (i * sC) % T
        base = truncated_hadamard(m, idx=(i*131 + 7))  # stabilní, „sequency“ variace
        # nastav přesný poměr -1 : +1 ~ zeta0
        k_neg = int(math.floor(zeta0 * m))
        signs = base.copy()
        neg = np.where(signs < 0)[0].tolist()
        pos = np.where(signs > 0)[0].tolist()
        rng.shuffle(neg); rng.shuffle(pos)
        need = k_neg - len(neg)
        if need > 0:
            for p in pos[:need]: signs[p] = -1
        elif need < 0:
            for p in neg[:(-need)]: signs[p] = +1

        for t in range(m):
            tt = (off + t) % T
            sgn = 1 if signs[t] > 0 else -1
            for lit in clause:
                j = abs(lit) - 1
                if j >= C: continue
                vshift = (j * sV) % T
                ttv = (tt + vshift) % T
                vote[ttv, j] += sgn

    if not two_pass:
        # single-pass: top-|vote| -> fáze
        phi  = np.zeros((T, C), dtype=np.float64)
        mask = np.zeros((T, C), dtype=np.float64)
        for j in range(C):
            col = vote[:, j]
            if np.all(col == 0): continue
            idxs = np.argsort(-np.abs(col))[:m]
            pol = 1 if np.sum(col[idxs]) >= 0 else -1
            for tt in idxs:
                bit = pol * np.sign(col[tt])
                phi[tt, j]  = 0.0 if bit >= 0 else np.pi
                mask[tt, j] = 1.0
        return phi, mask, T, m

    # two-pass: majorita -> jemný výběr
    major = np.ones(C, dtype=np.int8)
    for j in range(C):
        col = vote[:, j]
        if np.all(col == 0): continue
        major[j] = 1 if np.sum(np.sign(col)) >= 0 else -1

    phi  = np.zeros((T, C), dtype=np.float64)
    mask = np.zeros((T, C), dtype=np.float64)
    for j in range(C):
        col = vote[:, j]
        if np.all(col == 0): continue
        # preferuj časové indexy, kde znamení hlasů odpovídá majoritě
        pref = np.where(np.sign(col) * major[j] >= 0, 1.0, 0.5)
        score = pref * np.abs(col)
        idxs = np.argsort(-score)[:m]
        for tt in idxs:
            bit = major[j] * np.sign(col[tt])
            phi[tt, j]  = 0.0 if bit >= 0 else np.pi
            mask[tt, j] = 1.0
    return phi, mask, T, m

# --------------------- feedback ---------------------
def feedback_align(phi, mask, rounds=6, gate_start=0.15, gate_end=0.85, jitter=1e-6):
    """
    Vícekrokové fázové zarovnání s annealingem:
      - v kole r použij flip-gate τ_r lineárně od gate_start -> gate_end (přísnější s koly)
      - flipni jen ty fáze, kde cos(φ - mean_ang) < -τ_r
      - pokud |cos| velmi blízko 0, přidej miniaturní jitter (tie-break)
      - šumová brána: pokud energie sloupce je mizivá, flipy se nevykonají
    """
    T, C = phi.shape
    phi = phi.copy()
    mask = mask.copy()
    # předpočítej energii sloupců pro noise-gating
    energy = np.maximum(1e-12, mask.sum(axis=0) / float(T))

    for r in range(rounds):
        tau = gate_start + (gate_end - gate_start) * (r / max(1, rounds-1))
        for j in range(C):
            idxs = np.where(mask[:, j] > 0)[0]
            if idxs.size == 0: continue
            if energy[j] < 1e-4:  # noise gating
                continue
            v = np.exp(1j * phi[idxs, j])
            mvec = np.mean(v)
            if np.abs(mvec) < 1e-12:
                continue
            ang = np.angle(mvec)
            d = np.cos(phi[idxs, j] - ang)
            # tie-breaker lehké pošťouchnutí, když |d| ~ 0
            near0 = np.where(np.abs(d) < 1e-9)[0]
            if near0.size > 0:
                phi[idxs[near0], j] += (np.random.default_rng(j*(r+1)).standard_normal(near0.size) * jitter)
            # flip-gate: překlápěj jen dostatečně protifázové vzorky
            flips = np.where(d < -tau)[0]
            if flips.size:
                phi[idxs[flips], j] = (phi[idxs[flips], j] + np.pi) % (2*np.pi)
    return phi, mask

# ---------------------- invariants ----------------------
def von_mises_kappa(R):
    eps = 1e-12
    R = float(max(eps, min(1.0 - eps, R)))
    return (R*(2 - R**2)) / (1 - R**2)

def topk_entropy(P, K=7):
    """
    1 - H_topK/log K na spektru normalizovaném na sum(topK)=1 (stabilní škála).
    """
    if P.size == 0: return 0.0
    K = max(1, min(K, P.size))
    idx = np.argsort(P)[-K:]
    pk = P[idx]
    s = float(pk.sum())
    if s <= 0: return 0.0
    q = pk / s
    H = -np.sum(q * (np.log(q + 1e-12)))
    return 1.0 - float(H / np.log(K))

def features_from_phi(phi, mask, topK=7, align_alpha=1.0, sharp_beta=2.0):
    T, C = phi.shape
    sharp = np.zeros(C); coh = np.zeros(C); th = np.zeros(C)
    ent = []
    for j in range(C):
        z = (mask[:, j].astype(np.float64)) * np.exp(1j * phi[:, j])
        if np.allclose(z, 0):
            ent.append(0.0); continue
        Z = np.fft.fft(z)
        R = np.fft.ifft(np.abs(Z)**2).real
        P = np.abs(R[1:])  # line spectrum mimo lag 0
        if P.size == 0:
            ent.append(0.0); continue
        k = 1 + int(np.argmax(P))
        sharp[j] = float(P[k-1] / (np.sum(P) + 1e-12))   # „frakce“ hlavního píku
        coh[j]   = float(np.abs(np.mean(z)))
        th[j]    = 2*np.pi*k/T
        ent.append(topk_entropy(P, K=topK))
    w = (np.maximum(0, sharp)**sharp_beta) * (np.maximum(0, coh)**align_alpha)
    if np.sum(w) <= 0:
        align = 0.0; kappa = 0.0
    else:
        Rvec  = np.sum(w * np.exp(1j*th)) / (np.sum(w) + 1e-12)
        align = float(np.abs(Rvec))
        kappa = float(von_mises_kappa(align))

    Zstack = (mask * np.exp(1j*phi)).astype(complex)
    if C > 1:
        G = (Zstack.conj().T @ Zstack) / max(1, Zstack.shape[0])
        W = np.outer(w, w); np.fill_diagonal(W, 0.0)
        num = float(np.sum(W * np.abs(G))); den = float(np.sum(W) + 1e-12)
        Kc = num / den if den > 0 else 0.0
    else:
        Kc = 0.0
    mu_res = float(np.mean(sharp))
    E = float(np.mean(ent))
    return dict(mu_res=mu_res, align=align, kappa=kappa, K=Kc, E=E, coh=float(np.mean(coh)))

def score_geo(feats, use_E=True):
    base = feats['mu_res'] * feats['kappa'] * feats['K']
    return base * feats['E'] if use_E else base

# ---------------------- pipeline ----------------------
def _run_core(n, clauses, *, cR, rho, zeta0, L, seed, sC_frac, sV_frac, two_pass, fb_rounds, topK, align_alpha, sharp_beta, use_E=True):
    phi, mask, T, _ = schedule_instance_double(n, clauses, cR=cR, rho=rho, zeta0=zeta0, L=L, seed=seed,
                                               sC_frac=sC_frac, sV_frac=sV_frac, two_pass=two_pass)
    phi2, mask2 = feedback_align(phi, mask, rounds=fb_rounds)
    feats = features_from_phi(phi2, mask2, topK=topK, align_alpha=align_alpha, sharp_beta=sharp_beta)
    return feats, T

def classify_file(path, args):
    n, m, clauses = parse_dimacs(path)
    feats, T = _run_core(n, clauses,
                         cR=args.cR, rho=args.rho, zeta0=args.zeta0, L=args.L, seed=args.seed,
                         sC_frac=args.sC_frac, sV_frac=args.sV_frac, two_pass=args.two_pass,
                         fb_rounds=args.fb_rounds, topK=args.topK, align_alpha=args.align_alpha, sharp_beta=args.sharp_beta,
                         use_E=(not args.noE))
    s = score_geo(feats, use_E=(not args.noE))
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
        print("[bench] No positive gap — tune cR/rho/fb_rounds or strides.")

# ---------------------- tune/ensemble ----------------------
def tune_cmd(args):
    grid_cR = [args.cR] if not args.wide else [max(6, args.cR-4), args.cR, args.cR+4]
    grid_rho = [args.rho] if not args.wide else [max(0.5, args.rho-0.1), args.rho, min(0.95, args.rho+0.1)]
    grid_fb  = [args.fb_rounds] if not args.wide else [max(3, args.fb_rounds-2), args.fb_rounds, args.fb_rounds+2]
    grid_sC  = [args.sC_frac] if not args.wide else [args.sC_frac-0.02, args.sC_frac, args.sC_frac+0.02]
    grid_sV  = [args.sV_frac] if not args.wide else [args.sV_frac-0.04, args.sV_frac, args.sV_frac+0.04]

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
        print("no winner; try --wide")

def ensemble_cmd(args):
    stride_grid = [(args.sC_frac, args.sV_frac)]
    had_idxs = list(range(args.ensemble_masks))
    for p in args.paths:
        n,m,cla = parse_dimacs(p)
        acc = []
        for (sC,sV) in stride_grid:
            for h in had_idxs:
                feats, T = _run_core(n, cla,
                                     cR=args.cR, rho=args.rho, zeta0=args.zeta0, L=args.L, seed=args.seed+h*97,
                                     sC_frac=sC, sV_frac=sV, two_pass=args.two_pass,
                                     fb_rounds=args.fb_rounds, topK=args.topK,
                                     align_alpha=args.align_alpha, sharp_beta=args.sharp_beta,
                                     use_E=(not args.noE))
                acc.append(feats)
        keys = acc[0].keys()
        agg = {k: float(np.mean([a[k] for a in acc])) for k in keys}
        agg_score = score_geo(agg, use_E=(not args.noE))
        print(f"{Path(p).name:15} score={agg_score:.6f}  mu={agg['mu_res']:.4f}  coh={agg['coh']:.4f}  E={agg['E']:.4f}  kappa={agg['kappa']:.4f}  K={agg['K']:.4f}")

# -------------------- chaos & margin --------------------
def chaos_cmd(args):
    for p in args.paths:
        n,m,cla = parse_dimacs(p)
        phi, mask, T, _ = schedule_instance_double(n, cla,
            cR=args.cR, rho=args.rho, zeta0=args.zeta0, L=args.L, seed=args.seed,
            sC_frac=args.sC_frac, sV_frac=args.sV_frac, two_pass=args.two_pass
        )
        cohs = []
        ph = phi.copy()
        for r in range(args.fb_rounds):
            # jedna „ruční“ iterace feedbacku (gate roste)
            ph, _ = feedback_align(ph, mask, rounds=1, gate_start=0.15+r*(0.7/args.fb_rounds), gate_end=0.15+r*(0.7/args.fb_rounds))
            # koherence po kole
            C = ph.shape[1]
            col_coh = []
            for j in range(C):
                idxs = np.where(mask[:, j] > 0)[0]
                if idxs.size == 0: continue
                col_coh.append(float(np.abs(np.mean(np.exp(1j*ph[idxs, j])))))
            cohs.append(float(np.mean(col_coh)) if col_coh else 0.0)
        slope = float(np.polyfit(np.arange(1,len(cohs)+1), np.array(cohs), 1)[0]) if len(cohs)>=2 else 0.0
        feats = features_from_phi(ph, mask, topK=args.topK, align_alpha=args.align_alpha, sharp_beta=args.sharp_beta)
        print(f"{Path(p).name:15} coh_track={[round(x,4) for x in cohs]} slope={slope:.4f} score={score_geo(feats):.6f}")

def _circ_diff(a, b):  # signed angle in (-pi,pi]
    return (a - b + np.pi) % (2*np.pi) - np.pi

def margin_cmd(args):
    for p in args.paths:
        n,m,cla = parse_dimacs(p)
        phi, mask, T, _ = schedule_instance_double(n, cla,
            cR=args.cR, rho=args.rho, zeta0=args.zeta0, L=args.L, seed=args.seed,
            sC_frac=args.sC_frac, sV_frac=args.sV_frac, two_pass=args.two_pass
        )
        phi2, mask2 = feedback_align(phi, mask, rounds=args.fb_rounds)
        idxs = np.where(mask>0)
        bin0 = (phi[idxs] < (np.pi/2)).astype(int)
        bin1 = (phi2[idxs] < (np.pi/2)).astype(int)
        dH = np.abs(bin1 - bin0)
        dtheta = []
        for j in range(phi.shape[1]):
            jj = np.where(mask2[:, j] > 0)[0]
            if jj.size == 0: continue
            ang = np.angle(np.mean(np.exp(1j*phi2[jj, j])))
            dtheta.extend(np.abs(_circ_diff(phi2[jj, j], ang)).tolist())
        med_dH = float(np.median(dH)) if dH.size else 0.0
        med_dtheta = float(np.median(dtheta)) if dtheta else 0.0
        margin = med_dH - med_dtheta
        print(f"{Path(p).name:15} margin={margin:.6f}  med_dH={med_dH:.4f}  med_dtheta={med_dtheta:.4f}")

# -------------------------- CLI --------------------------
def build_base():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("paths", nargs="+", help="CNF files")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cR", type=float, default=12.0)
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--zeta0", type=float, default=0.4)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--fb_rounds", type=int, default=6)
    ap.add_argument("--sC_frac", type=float, default=0.47)
    ap.add_argument("--sV_frac", type=float, default=0.31)
    ap.add_argument("--topK", type=int, default=9)
    ap.add_argument("--align_alpha", type=float, default=1.0)
    ap.add_argument("--sharp_beta", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--noE", action="store_true")
    ap.add_argument("--two_pass", action="store_true")
    return ap

def main():
    root = argparse.ArgumentParser(description="AO GEO Resonance — full.")
    sub = root.add_subparsers(dest="cmd")
    base = build_base()
    sub.add_parser("classify", parents=[base])
    sub.add_parser("bench", parents=[base])
    t = sub.add_parser("tune", parents=[base]); t.add_argument("--wide", action="store_true")
    e = sub.add_parser("ensemble", parents=[base]); e.add_argument("--ensemble_masks", type=int, default=5)
    sub.add_parser("chaos", parents=[base])
    sub.add_parser("margin", parents=[base])
    args = root.parse_args()
    if args.cmd == "classify": classify_cmd(args)
    elif args.cmd == "bench":  bench_cmd(args)
    elif args.cmd == "tune":   tune_cmd(args)
    elif args.cmd == "ensemble": ensemble_cmd(args)
    elif args.cmd == "chaos":  chaos_cmd(args)
    elif args.cmd == "margin": margin_cmd(args)
    else:
        if hasattr(args, "paths"): classify_cmd(args)
        else: root.print_help()

if __name__ == "__main__":
    main()
