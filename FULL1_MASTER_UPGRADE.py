#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL VERSION: AO_GEO_resonance_full.py
Pure Geometry Resonance — sjednocená a kompletní verze:
- DIMACS CNF parsing
- Schedules:
    * lock-only double coprime stride (T = L * R, R = ceil(cR * log max(2,n)))
    * two-pass varianta (globální polarita -> jemné rozmístění)
- Walsh–Hadamard (Sylvester) generátor s Walsh/sequency pořadím + bezpečný ořez
- Geometrický feedback s annealingem, flip-gate, tie-breaker jitterem a noise gatingem
- Invarianta z kruhové autokorelace: mu_res, coh, top-K entropie E (1=peaky),
  lag-alignment align, von-Mises kappa, cross-column K
- Skóre: score = mu_res * kappa * K * E  (E lze vypnout)
- Subcommands: classify, bench, tune, ensemble, chaos, margin
"""
import argparse, math, statistics, time, random
from pathlib import Path
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt


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
                    n = int(parts[-2]);
                    m = int(parts[-1])
                continue
            xs = [int(x) for x in s.split() if x]
            if xs and xs[-1] == 0: xs.pop()
            if xs: clauses.append(tuple(xs))
    return n, len(clauses), clauses


# -------------------------- helpers --------------------------
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)


def _coprime(a, b):
    return _gcd(a, b) == 1


def stride_near(T: int, frac: float, forbid=(1, 2), search_radius=None):
    if T <= 4:
        return max(1, T - 2)
    target = int(round((frac % 1.0) * T)) % T
    if target < 2: target = 2
    if target > T - 2: target = T - 2
    R = search_radius or max(8, int(0.15 * T))

    def score(s: int) -> float:
        triv = {1, 2, T - 1, T - 2}
        pen = 3.0 if s in triv or s in forbid else 0.0
        alias = 0.0
        for d in range(2, min(32, T // 2) + 1):
            if T % d == 0:
                step = T // d
                k = round(s / step)
                alias += abs(s - k * step) / step if abs(s - k * step) < 0.5 else 0.0
        return abs(s - target) + pen + 0.1 * alias

    candidates = [s for s in range(2, T - 1) if _coprime(s, T)]
    return min(candidates, key=score) if candidates else 2


def truncated_hadamard(m: int, idx: int = 1):
    if m <= 1: return np.array([1], dtype=np.int8)
    baseN = 1
    while baseN < m: baseN <<= 1
    rng = np.random.default_rng(idx)
    row = np.ones(baseN, dtype=np.int8)
    for i in range(baseN // 2):
        j = rng.integers(0, baseN)
        row[j] = -1 if row[j] == 1 else 1
    return row[:m]


# -------------------------- schedules --------------------------
def _ensure_distinct_coprime_strides(T, sC, sV):
    if not _coprime(sC, T): sC = stride_near(T, sC / T + 1e-9)
    if not _coprime(sV, T): sV = stride_near(T, sV / T + 2e-9)
    if sC == sV:
        sV = stride_near(T, (sV + 1) / T)
    if not _coprime(sC, sV):
        sV = stride_near(T, (sV + 3) / T)
    return sC, sV


def schedule_instance_double(n, clauses, cR=12.0, rho=0.6, zeta0=0.4, L=3, seed=42,
                             sC_frac=0.47, sV_frac=0.31, two_pass=False):
    rng = np.random.default_rng(seed)
    C = max(1, n)
    R = max(1, int(math.ceil(cR * math.log(max(2, C)))))
    T = int(L * R)
    m = max(1, int(math.floor(rho * T)))

    sC = stride_near(T, sC_frac)
    sV = stride_near(T, sV_frac)
    sC, sV = _ensure_distinct_coprime_strides(T, sC, sV)

    vote = np.zeros((T, C), dtype=np.int32)
    for i, clause in enumerate(clauses):
        off = (i * sC) % T
        base = truncated_hadamard(m, idx=(i * 131 + 7))
        k_neg = int(math.floor(zeta0 * m))
        signs = base.copy()
        neg = np.where(signs < 0)[0].tolist()
        pos = np.where(signs > 0)[0].tolist()
        rng.shuffle(neg);
        rng.shuffle(pos)
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
        phi = np.zeros((T, C), dtype=np.float64)
        mask = np.zeros((T, C), dtype=np.float64)
        for j in range(C):
            col = vote[:, j]
            if np.all(col == 0): continue
            idxs = np.argsort(-np.abs(col))[:m]
            pol = 1 if np.sum(col[idxs]) >= 0 else -1
            for tt in idxs:
                bit = pol * np.sign(col[tt])
                phi[tt, j] = 0.0 if bit >= 0 else np.pi
                mask[tt, j] = 1.0
        return phi, mask, T, m

    major = np.ones(C, dtype=np.int8)
    for j in range(C):
        col = vote[:, j]
        if np.all(col == 0): continue
        major[j] = 1 if np.sum(np.sign(col)) >= 0 else -1

    phi = np.zeros((T, C), dtype=np.float64)
    mask = np.zeros((T, C), dtype=np.float64)
    for j in range(C):
        col = vote[:, j]
        if np.all(col == 0): continue
        pref = np.where(np.sign(col) * major[j] >= 0, 1.0, 0.5)
        score = pref * np.abs(col)
        idxs = np.argsort(-score)[:m]
        for tt in idxs:
            bit = major[j] * np.sign(col[tt])
            phi[tt, j] = 0.0 if bit >= 0 else np.pi
            mask[tt, j] = 1.0
    return phi, mask, T, m


def schedule_instance_two_pass(n, clauses, **kw):
    kw['two_pass'] = True
    return schedule_instance_double(n, clauses, **kw)


# -------------------------- feedback --------------------------
def feedback_align(phi, mask, rounds=6, gate_start=0.15, gate_end=0.85, jitter=1e-6, track=False):
    T, C = phi.shape
    phi = phi.copy()
    mask = mask.copy()
    energy = np.maximum(1e-12, mask.sum(axis=0) / float(T))
    cohs = []

    for r in range(rounds):
        tau = gate_start + (gate_end - gate_start) * (r / max(1, rounds - 1))
        for j in range(C):
            idxs = np.where(mask[:, j] > 0)[0]
            if idxs.size == 0: continue
            if energy[j] < 1e-4: continue
            v = np.exp(1j * phi[idxs, j])
            mvec = np.mean(v)
            if np.abs(mvec) < 1e-12: continue

            ang = np.angle(mvec)
            d = np.cos(phi[idxs, j] - ang)
            near0 = np.where(np.abs(d) < 1e-9)[0]
            if near0.size > 0:
                phi[idxs[near0], j] += (np.random.default_rng(j * (r + 1)).standard_normal(near0.size) * jitter)
            flips = np.where(d < -tau)[0]
            if flips.size:
                phi[idxs[flips], j] = (phi[idxs[flips], j] + np.pi) % (2 * np.pi)

        if track:
            col_coh = []
            for j in range(C):
                idxs = np.where(mask[:, j] > 0)[0]
                if idxs.size == 0: continue
                col_coh.append(float(np.abs(np.mean(np.exp(1j * phi[idxs, j])))))
            cohs.append(float(np.mean(col_coh)) if col_coh else 0.0)
    return (phi, mask, cohs) if track else (phi, mask)


# -------------------------- invariants --------------------------
def von_mises_kappa(R):
    eps = 1e-12
    R = float(max(eps, min(1.0 - eps, R)))
    return (R * (2 - R ** 2)) / (1 - R ** 2)


def topk_entropy(P, K=7):
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
    sharp = np.zeros(C);
    coh = np.zeros(C);
    th = np.zeros(C)
    ent = []
    for j in range(C):
        z = (mask[:, j].astype(np.float64)) * np.exp(1j * phi[:, j])
        if np.allclose(z, 0):
            ent.append(0.0);
            continue
        Z = np.fft.fft(z)
        R = np.fft.ifft(np.abs(Z) ** 2).real
        P = np.abs(R[1:])
        if P.size == 0:
            ent.append(0.0);
            continue
        k = 1 + int(np.argmax(P))
        sharp[j] = float(P[k - 1] / (np.sum(P) + 1e-12))
        coh[j] = float(np.abs(np.mean(z)))
        th[j] = 2 * np.pi * k / T
        ent.append(topk_entropy(P, K=topK))
    w = (np.maximum(0, sharp) ** sharp_beta) * (np.maximum(0, coh) ** align_alpha)
    if np.sum(w) <= 0:
        align = 0.0;
        kappa = 0.0
    else:
        Rvec = np.sum(w * np.exp(1j * th)) / (np.sum(w) + 1e-12)
        align = float(np.abs(Rvec))
        kappa = float(von_mises_kappa(align))
    Zstack = (mask * np.exp(1j * phi)).astype(complex)
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
    E = float(np.mean(ent))
    return dict(mu_res=mu_res, align=align, kappa=kappa, K=Kc, E=E, coh=float(np.mean(coh)))


def score_geo(feats, use_E=True):
    base = feats['mu_res'] * feats['kappa'] * feats['K']
    return base * feats['E'] if use_E else base


# -------------------------- pipeline --------------------------
def _run_core(n, clauses, *, cR, rho, zeta0, L, seed, sC_frac, sV_frac, two_pass, fb_rounds, topK, align_alpha,
              sharp_beta, use_E=True, track_coh=False):
    phi, mask, T, m = schedule_instance_double(n, clauses, cR=cR, rho=rho, zeta0=zeta0, L=L, seed=seed,
                                               sC_frac=sC_frac, sV_frac=sV_frac, two_pass=two_pass)
    ret = feedback_align(phi, mask, rounds=fb_rounds, track=track_coh)
    phi2, mask2 = ret[0], ret[1]
    coh_track = ret[2] if track_coh else None
    feats = features_from_phi(phi2, mask2, topK=topK, align_alpha=align_alpha, sharp_beta=sharp_beta)
    return feats, T, coh_track


def classify_file(path, args, cfg):
    n, m, clauses = parse_dimacs(path)
    feats, T, _ = _run_core(n, clauses, **cfg, topK=args.topK, align_alpha=args.align_alpha, sharp_beta=args.sharp_beta,
                            use_E=(not args.noE), track_coh=False)
    s = score_geo(feats, use_E=(not args.noE))
    label = "SAT-like" if s >= args.tau else "UNSAT-like"
    return {"file": Path(path).name, "n": n, "m": m, "T": T, **feats, "score": s, "label": label}


def classify_cmd(args):
    cfg = build_cfg(args)
    rows = [classify_file(p, args, cfg) for p in args.paths]
    print(
        f"{'file':15} {'n':>3} {'m':>4} {'T':>4} {'mu_res':>8} {'align':>8} {'coh':>8} {'E':>8} {'kappa':>8} {'K':>8} {'score':>10} {'label':>10}")
    for r in rows:
        print(f"{r['file']:15} {r['n']:3d} {r['m']:4d} {r['T']:4d} "
              f"{r['mu_res']:8.4f} {r['align']:8.4f} {r['coh']:8.4f} {r['E']:8.4f} "
              f"{r['kappa']:8.4f} {r['K']:8.4f} {r['score']:10.6f} {r['label']:>10}")
    return rows


def bench_cmd(args):
    cfg = build_cfg(args)
    rows = classify_cmd(args)
    uf = [r for r in rows if r["file"].startswith("uf")]
    uuf = [r for r in rows if r["file"].startswith("uuf")]
    if not uf or not uuf:
        print("\n[bench] Need both uf* and uuf* inputs for threshold computation.")
        return
    uf_scores = sorted([r["score"] for r in uf])
    uuf_scores = sorted([r["score"] for r in uuf])
    min_uf = min(uf_scores);
    max_uuf = max(uuf_scores)
    gap = min_uf - max_uuf
    tau_star = 0.5 * (min_uf + max_uuf)
    print("\n[bench] uf scores :", [f"{x:.6f}" for x in uf_scores])
    print("[bench] uuf scores:", [f"{x:.6f}" for x in uuf_scores])
    print(f"[bench] min(uf)={min_uf:.6f}, max(uuf)={max_uuf:.6f}, gap={gap:.6f}")
    if gap > 0:
        print(f"[bench] Suggested τ* = {tau_star:.6f} (separates uf from uuf on this set)")
    else:
        print("[bench] No positive gap — tune cR/rho/fb_rounds or strides.")


# ---------------------- tune/ensemble ----------------------
def tune_cmd(args):
    grid_cR = [args.cR] if not args.wide else [max(6, args.cR - 4), args.cR, args.cR + 4]
    grid_rho = [args.rho] if not args.wide else [max(0.5, args.rho - 0.1), args.rho, min(0.95, args.rho + 0.1)]
    grid_fb = [args.fb_rounds] if not args.wide else [max(3, args.fb_rounds - 2), args.fb_rounds, args.fb_rounds + 2]
    grid_sC = [args.sC_frac] if not args.wide else [args.sC_frac - 0.02, args.sC_frac, args.sC_frac + 0.02]
    grid_sV = [args.sV_frac] if not args.wide else [args.sV_frac - 0.04, args.sV_frac, args.sV_frac + 0.04]

    best = None
    print(f"{'cR':>4} {'rho':>4} {'fb':>3} {'sC':>5} {'sV':>5} {'min_uf':>8} {'max_uuf':>8} {'gap':>8} {'thr':>8}")
    for cR in grid_cR:
        for rho in grid_rho:
            for fb in grid_fb:
                for sC in grid_sC:
                    for sV in grid_sV:
                        ns = argparse.Namespace(**vars(args))
                        ns.cR, ns.rho, ns.fb_rounds, ns.sC_frac, ns.sV_frac = float(cR), float(rho), int(fb), float(
                            sC), float(sV)
                        rows = [classify_file(p, ns, build_cfg(ns)) for p in ns.paths]
                        uf = [r for r in rows if r["file"].startswith("uf")]
                        uuf = [r for r in rows if r["file"].startswith("uuf")]
                        if not uf or not uuf: continue
                        uf_scores = sorted([r["score"] for r in uf])
                        uuf_scores = sorted([r["score"] for r in uuf])
                        min_uf = min(uf_scores);
                        max_uuf = max(uuf_scores)
                        gap = min_uf - max_uuf
                        thr = 0.5 * (min_uf + max_uuf)
                        print(
                            f"{cR:4.1f} {rho:4.1f} {fb:3d} {sC:5.2f} {sV:5.2f} {min_uf:8.4f} {max_uuf:8.4f} {gap:8.5f} {thr:8.5f}")
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
        n, m, cla = parse_dimacs(p)
        acc = []
        for (sC, sV) in stride_grid:
            for h in had_idxs:
                cfg = build_cfg(args)
                cfg["sC_frac"] = sC;
                cfg["sV_frac"] = sV;
                cfg["seed"] = args.seed + h * 97
                feats, T, _ = _run_core(n, cla, **cfg, topK=args.topK, align_alpha=args.align_alpha,
                                        sharp_beta=args.sharp_beta, use_E=(not args.noE), track_coh=False)
                acc.append(feats)
        keys = acc[0].keys()
        agg = {k: float(np.mean([a[k] for a in acc])) for k in keys}
        agg_score = score_geo(agg, use_E=(not args.noE))
        print(
            f"{Path(p).name:15} score={agg_score:.6f}  mu={agg['mu_res']:.4f}  coh={agg['coh']:.4f}  E={agg['E']:.4f}  kappa={agg['kappa']:.4f}  K={agg['K']:.4f}")


def chaos_cmd(args):
    for p in args.paths:
        n, m, cla = parse_dimacs(p)
        cfg = build_cfg(args)
        feats, T, coh_track = _run_core(n, cla, **cfg, topK=args.topK, align_alpha=args.align_alpha,
                                        sharp_beta=args.sharp_beta, use_E=(not args.noE), track_coh=True)
        s = score_geo(feats, use_E=not args.noE)
        slope = float(np.polyfit(np.arange(1, len(coh_track) + 1), np.array(coh_track), 1)[0]) if len(
            coh_track) >= 2 else 0.0
        print(f"{Path(p).name:15} coh_track={[round(x, 4) for x in coh_track]} slope={slope:.4f} score={s:.6f}")


def _circ_diff(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def margin_cmd(args):
    for p in args.paths:
        n, m, cla = parse_dimacs(p)
        cfg = build_cfg(args)
        phi, mask, T, _ = schedule_instance_double(n, cla, **cfg, two_pass=args.two_pass)
        phi2, mask2 = feedback_align(phi, mask, rounds=args.fb_rounds)
        idxs = np.where(mask > 0)
        bin0 = (phi[idxs] < (np.pi / 2)).astype(int)
        bin1 = (phi2[idxs] < (np.pi / 2)).astype(int)
        dH = np.abs(bin1 - bin0)
        dtheta = []
        for j in range(phi.shape[1]):
            jj = np.where(mask2[:, j] > 0)[0]
            if jj.size == 0: continue
            ang = np.angle(np.mean(np.exp(1j * phi2[jj, j])))
            dtheta.extend(np.abs(_circ_diff(phi2[jj, j], ang)).tolist())
        med_dH = float(np.median(dH)) if dH.size else 0.0
        med_dtheta = float(np.median(dtheta)) if dtheta else 0.0
        margin = med_dH - med_dtheta
        print(f"{Path(p).name:15} margin={margin:.6f}  med_dH={med_dH:.4f}  med_dtheta={med_dtheta:.4f}")


# -------------------------- CLI --------------------------
def build_base_parser():
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
    ap.add_argument("--noE", action="store_true", help="exclude E from score (debugging)")
    ap.add_argument("--two_pass", action="store_true", help="use two-pass schedule")
    return ap


def build_cfg(args):
    return dict(
        cR=args.cR, rho=args.rho, zeta0=args.zeta0, L=args.L, seed=args.seed,
        sC_frac=args.sC_frac, sV_frac=args.sV_frac, two_pass=args.two_pass
    )


def main():
    root = argparse.ArgumentParser(description="AO GEO Resonance — full.")
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
        if hasattr(args, "paths"):
            classify_cmd(args)
        else:
            root.print_help()


if __name__ == "__main__":
    main()