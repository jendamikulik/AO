#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REPORT_PNP.py
Deterministic CNF-driven spectral tester with batch evaluation, ROC, tau selection,
histograms, optional lock-only S2, and markdown report.

Usage:
  python REPORT_PNP.py --sat_dir path/to/SAT --unsat_dir path/to/UNSAT \
      --out_dir ./report_out --zeta0 0.40 --rho_lock 0.50 --cR 15.0 --col_stride 3 \
      --lock_only_s2 false

Author: AO pipeline (deterministic), 2025-09-20
"""

import os, sys, math, json, argparse, glob, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- DIMACS --------------------

def parse_dimacs(path):
    n_vars = None
    clauses = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln[0] in "c%":
                continue
            if ln.startswith("p cnf"):
                _,_,nv,mc = ln.split()
                n_vars, m_clauses = int(nv), int(mc)
            else:
                toks = ln.split()
                if toks and toks[-1] == '0':
                    lits = [int(x) for x in toks[:-1]]
                    if lits:
                        clauses.append(lits)
    if n_vars is None and clauses:
        n_vars = max(abs(l) for cls in clauses for l in cls)
    return n_vars or 0, clauses

# -------------------- Deterministic math helpers --------------------

def next_pow2(n):
    v = 1
    while v < n: v <<= 1
    return v

def sylvester_hadamard(n):
    # n must be power of 2
    H = np.array([[1]], dtype=np.int8)
    while H.shape[0] < n:
        H = np.block([[H, H],[H, -H]])
    return H

def gcd(a,b):
    while b: a,b=b,a%b
    return a

def coprime_stride(n):
    # choose odd stride near n//2, coprime with n (deterministic scan)
    center = max(1, n//2)
    for delta in range(0, n):
        for sign in (1,-1):
            s = (center + sign*delta)
            if s <= 0: continue
            if s % 2 == 0: continue
            if gcd(s, n) == 1:
                return s
    return 1

def dealiased_offsets(C, T):
    s = coprime_stride(T)
    offs = [(j*s) % T for j in range(C)]
    return offs, s

def var_row(i, H_len):
    # deterministic, odd-step mapping of variables to Hadamard rows
    return (1 + 3*i) % H_len

# -------------------- CNF → Φ schedule --------------------

def build_phi_from_cnf(T, m, clauses, zeta0, col_stride=3):
    """
    T time slots, lock length m, clauses = list of lists of signed ints.
    Deterministic Hadamard truncation and de-aliased offsets per clause.
    """
    C = len(clauses)
    Phi = np.full((T, C), np.pi, dtype=np.float64)

    H_len = next_pow2(m)
    H = sylvester_hadamard(H_len)

    offsets, stride_T = dealiased_offsets(C, T)
    col_stride = col_stride if col_stride % 2 == 1 else (col_stride+1)
    sub_idx = [(k*col_stride) % H_len for k in range(m)]
    kk = int(max(0, min(m, round(zeta0*m))))

    # Persist lock windows for optional lock-only S2
    lock_windows = []

    for j, lits in enumerate(clauses):
        agg = np.zeros(m, dtype=np.int32)
        for lit in lits:
            v = abs(lit)
            r = var_row(v, H_len)
            row = H[r, sub_idx].copy()
            if lit < 0:
                row *= -1
            agg += row
        order = np.argsort(agg)  # most negative first
        neg_set = set(order[:kk])
        start = offsets[j]
        lw = [(start + k) % T for k in range(m)]
        lock_windows.append(set(lw))
        for k in range(m):
            t = (start + k) % T
            Phi[t, j] = np.pi if k in neg_set else 0.0

    return Phi, lock_windows

# -------------------- Spectral metrics --------------------

def spectral_mu(Phi, power_iter=80):
    Z = np.exp(1j*Phi)       # T x C
    T = Phi.shape[0]
    G = (Z.conj().T @ Z) / T # C x C Hermitian
    C = G.shape[0]
    v = np.ones((C,), dtype=complex)
    v /= np.linalg.norm(v)
    for _ in range(power_iter):
        w = G @ v
        nw = np.linalg.norm(w)
        if nw == 0: break
        v = w / nw
    lam = float(np.real(np.vdot(v, G @ v)))  # Rayleigh
    mu = lam / C
    return mu, lam, G

def s2_dense(G):
    A = np.abs(G)
    mask = np.ones_like(A, dtype=bool)
    np.fill_diagonal(mask, 0)
    vals = A[mask]
    row_sums = (A - np.diag(np.diag(A))).sum(axis=1)
    return dict(
        abs_edge_max = float(np.max(vals)) if vals.size else 0.0,
        abs_edge_avg = float(np.mean(vals)) if vals.size else 0.0,
        row_sum_max  = float(np.max(row_sums)) if row_sums.size else 0.0,
        row_sum_avg  = float(np.mean(row_sums)) if row_sums.size else 0.0
    )

def lock_only_s2(Phi, lock_windows, sample_edges=2000, rng=np.random):
    """
    Estimate lock-only S2 by sampling pairs (i,j) and
    computing overlap-average inner product over intersect(Li, Lj).
    """
    T, C = Phi.shape
    pairs = []
    for _ in range(sample_edges):
        i = rng.randint(0, C)
        j = rng.randint(0, C)
        if i == j: continue
        pairs.append((i,j))
    Z = np.exp(1j*Phi)
    mags = []
    row_sum = np.zeros(C, dtype=float)
    for (i,j) in pairs:
        Sij = lock_windows[i].intersection(lock_windows[j])
        if not Sij:
            val = 0.0
        else:
            idx = np.fromiter(Sij, dtype=int)
            val = np.vdot(Z[idx, i], Z[idx, j]) / len(idx)
            val = abs(val)
        mags.append(val)
        row_sum[i] += val
        row_sum[j] += val
    mags = np.array(mags, dtype=float)
    return dict(
        abs_edge_max = float(np.max(mags)) if mags.size else 0.0,
        abs_edge_avg = float(np.mean(mags)) if mags.size else 0.0,
        row_sum_max  = float(np.max(row_sum)) if row_sum.size else 0.0,
        row_sum_avg  = float(np.mean(row_sum)) if row_sum.size else 0.0,
        sampled_edges = len(pairs)
    )

# -------------------- Batch, ROC, stats --------------------

def evaluate_dir(dirpath, label, T, m, zeta0, col_stride):
    rows = []
    files = sorted(glob.glob(os.path.join(dirpath, "*.cnf")))
    for fp in files:
        try:
            n_vars, clauses = parse_dimacs(fp)
            C = len(clauses)
            if C == 0:
                continue
            Phi, locks = build_phi_from_cnf(T, m, clauses, zeta0, col_stride)
            mu, lam, G = spectral_mu(Phi)
            s2g = s2_dense(G)
            rows.append(dict(
                file=os.path.basename(fp), path=fp, label=label,
                n_vars=n_vars, C=C, mu=mu, lambda_max=lam,
                s2_edge_max=s2g["abs_edge_max"],
                s2_edge_avg=s2g["abs_edge_avg"],
                s2_row_max=s2g["row_sum_max"],
                s2_row_avg=s2g["row_sum_avg"]
            ))
        except Exception as e:
            rows.append(dict(file=os.path.basename(fp), path=fp, label=label, error=str(e)))
    return rows

def compute_R_T(C, cR):
    R = max(3, int(round(cR * math.log(max(C,2)))))
    T = 3*R
    return R, T

def pick_tau_mu(rows):
    # thresholds from union of mu values
    vals = sorted(set(round(r["mu"], 12) for r in rows if "mu" in r))
    if not vals: return 0.5, 0.0, (0,0,0,0), 0.0
    # labels: SAT=1, UNSAT=0
    ys = np.array([1 if r["label"]=="SAT" else 0 for r in rows if "mu" in r], dtype=int)
    mus = np.array([r["mu"] for r in rows if "mu" in r], dtype=float)
    # ROC and Youden J
    best_tau, best_J, best_tuple = None, -1.0, None
    tprs, fprs = [], []
    for tau in vals:
        yhat = (mus >= tau).astype(int)
        TP = int(np.sum((yhat==1)&(ys==1)))
        FP = int(np.sum((yhat==1)&(ys==0)))
        TN = int(np.sum((yhat==0)&(ys==0)))
        FN = int(np.sum((yhat==0)&(ys==1)))
        P = max(1, TP+FN); N = max(1, TN+FP)
        TPR = TP/P; FPR = FP/N
        J = TPR - FPR
        tprs.append(TPR); fprs.append(FPR)
        if J > best_J:
            best_J, best_tau, best_tuple = J, tau, (TP,FP,TN,FN)
    # AUC (simple trapezoid on sorted FPR)
    idx = np.argsort(fprs)
    auc = 0.0
    for i in range(1, len(idx)):
        x0, x1 = fprs[idx[i-1]], fprs[idx[i]]
        y0, y1 = tprs[idx[i-1]], tprs[idx[i]]
        auc += 0.5*(y0+y1)*(x1-x0)
    return best_tau, auc, best_tuple, best_J

def cohens_d(x, y):
    x = np.array(x, dtype=float); y = np.array(y, dtype=float)
    mx, my = x.mean(), y.mean()
    nx, ny = len(x), len(y)
    sx2 = np.var(x, ddof=1) if nx>1 else 0.0
    sy2 = np.var(y, ddof=1) if ny>1 else 0.0
    s2p = ((nx-1)*sx2 + (ny-1)*sy2) / max(1,(nx+ny-2))
    sp = math.sqrt(max(0.0, s2p))
    return 0.0 if sp==0 else (mx-my)/sp

def write_csv(rows, path):
    if not rows: return
    keys = sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys)+"\n")
        for r in rows:
            f.write(",".join(str(r.get(k,"")) for k in keys)+"\n")

# -------------------- Plotting --------------------

def plot_hist_mu(rows, out_png):
    mus_sat = [r["mu"] for r in rows if "mu" in r and r["label"]=="SAT"]
    mus_uns = [r["mu"] for r in rows if "mu" in r and r["label"]=="UNSAT"]
    plt.figure(figsize=(7,4))
    bins = np.linspace(min(mus_sat+mus_uns), max(mus_sat+mus_uns), 40) if mus_sat and mus_uns else 30
    plt.hist(mus_sat, bins=bins, alpha=0.6, label="SAT", color="#2ca02c")
    plt.hist(mus_uns, bins=bins, alpha=0.6, label="UNSAT", color="#d62728")
    plt.xlabel("μ = λ_max / C")
    plt.ylabel("count")
    plt.title("Histogram of μ (SAT vs UNSAT)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_roc(rows, tau, out_png):
    ys = np.array([1 if r["label"]=="SAT" else 0 for r in rows if "mu" in r], dtype=int)
    mus = np.array([r["mu"] for r in rows if "mu" in r], dtype=float)
    ths = sorted(set(mus))
    pts = []
    for t in ths:
        yhat = (mus >= t).astype(int)
        TP = int(np.sum((yhat==1)&(ys==1)))
        FP = int(np.sum((yhat==1)&(ys==0)))
        TN = int(np.sum((yhat==0)&(ys==0)))
        FN = int(np.sum((yhat==0)&(ys==1)))
        P = max(1, TP+FN); N = max(1, TN+FP)
        TPR = TP/P; FPR = FP/N
        pts.append((FPR, TPR))
    pts = sorted(pts)
    plt.figure(figsize=(5,5))
    xs = [p[0] for p in pts]; ys2 = [p[1] for p in pts]
    plt.plot(xs, ys2, '-o', ms=2, label="ROC (μ)")
    plt.plot([0,1],[0,1],'k--',alpha=0.3)
    plt.scatter([None],[None])  # placeholder
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC for μ; τ*={tau:.4f}")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_mu_vs_s2(rows, out_png):
    mus = [r["mu"] for r in rows if "mu" in r]
    s2a = [r["s2_row_avg"] if "s2_row_avg" in r else r.get("s2_row_avg","") for r in rows if "mu" in r]
    labs = [r["label"] for r in rows if "mu" in r]
    col = ["#2ca02c" if L=="SAT" else "#d62728" for L in labs]
    plt.figure(figsize=(6.5,4))
    plt.scatter(s2a, mus, c=col, alpha=0.7, edgecolors='k', linewidths=0.3)
    plt.xlabel("row_sum_avg (|G|)") ; plt.ylabel("μ")
    plt.title("μ vs S2 row_sum_avg")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sat_dir", required=True)
    ap.add_argument("--unsat_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cR", type=float, default=15.0)
    ap.add_argument("--rho_lock", type=float, default=0.50)
    ap.add_argument("--zeta0", type=float, default=0.40)
    ap.add_argument("--col_stride", type=int, default=3)
    ap.add_argument("--lock_only_s2", type=str, default="false")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # Warm up on a small phantom C to get R,T; then per-file T is fixed by C via same formula
    # For batch consistency we choose a single (R,T) based on median C across both dirs.
    all_c = []
    for d in (args.sat_dir, args.unsat_dir):
        for fp in glob.glob(os.path.join(d, "*.cnf")):
            _, cls = parse_dimacs(fp)
            all_c.append(len(cls))
    if not all_c:
        print("No CNF found.", file=sys.stderr); sys.exit(2)
    C_med = int(np.median([c for c in all_c if c>0])) or max(all_c)
    R, T = compute_R_T(C_med, args.cR)
    m = int(round(args.rho_lock * T))

    rows_sat = evaluate_dir(args.sat_dir, "SAT", T, m, args.zeta0, args.col_stride)
    rows_uns = evaluate_dir(args.unsat_dir, "UNSAT", T, m, args.zeta0, args.col_stride)
    rows = rows_sat + rows_uns

    # Optional lock-only S2 (subset for speed)
    if args.lock_only_s2.lower() in ("true","1","yes","y"):
        # Recompute Φ + locks for some files to estimate lock-only S2
        sample = [r for r in rows if "mu" in r][: min(12, len(rows))]
        for r in sample:
            _, cls = parse_dimacs(r["path"])
            Phi, locks = build_phi_from_cnf(T, m, cls, args.zeta0, args.col_stride)
            r["s2_lock"] = lock_only_s2(Phi, locks, sample_edges=1500, rng=rng)

    # Write raw data
    write_csv(rows, os.path.join(args.out_dir, "rows.csv"))
    with open(os.path.join(args.out_dir, "rows.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # Split mus
    mus_sat = [r["mu"] for r in rows if "mu" in r and r["label"]=="SAT"]
    mus_uns = [r["mu"] for r in rows if "mu" in r and r["label"]=="UNSAT"]

    # ROC, tau, AUC, confusion, effect size
    tau, auc, (TP,FP,TN,FN), J = pick_tau_mu(rows)
    d_mu = cohens_d(np.array(mus_sat), np.array(mus_uns))

    # Plots
    plot_hist_mu(rows, os.path.join(args.out_dir, "hist_mu.png"))
    plot_roc(rows, tau, os.path.join(args.out_dir, "roc_mu.png"))
    plot_mu_vs_s2(rows, os.path.join(args.out_dir, "mu_vs_s2.png"))

    # Markdown report
    md = []
    md.append("# AO Spectral SAT Report\n")
    md.append(f"- SAT dir: `{args.sat_dir}`\n- UNSAT dir: `{args.unsat_dir}`\n")
    md.append(f"- Parameters: cR={args.cR}, R={R}, T={T}, m={m}, rho_lock={args.rho_lock}, zeta0={args.zeta0}, col_stride={args.col_stride}\n")
    md.append("\n## Summary\n")
    md.append(f"- Instances: SAT={len(rows_sat)}, UNSAT={len(rows_uns)}\n")
    md.append(f"- μ (mean ± std): SAT={np.mean(mus_sat):.4f} ± {np.std(mus_sat):.4f} | UNSAT={np.mean(mus_uns):.4f} ± {np.std(mus_uns):.4f}\n")
    md.append(f"- Cohen d(μ)={d_mu:.3f}\n")
    md.append(f"- ROC AUC(μ)={auc:.4f}, Youden J*={J:.4f}, τ*={tau:.6f}\n")
    md.append(f"- Confusion @ τ*: TP={TP}, FP={FP}, TN={TN}, FN={FN}\n")
    md.append("\n### Figures\n")
    md.append("![](hist_mu.png)\n\n")
    md.append("![](roc_mu.png)\n\n")
    md.append("![](mu_vs_s2.png)\n\n")
    if any("s2_lock" in r for r in rows):
        md.append("### Lock-only S2 (sample)\n")
        for r in rows:
            if "s2_lock" in r:
                s = r["s2_lock"]
                md.append(f"- `{os.path.basename(r['path'])}`: lock-S2 avg={s['abs_edge_avg']:.4f}, row_sum_avg={s['row_sum_avg']:.4f} (edges={s['sampled_edges']})\n")
    with open(os.path.join(args.out_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # Print console summary
    print(f"[OK] Rows: {len(rows)}; τ*={tau:.6f}; AUC={auc:.4f}; Confusion (TP,FP,TN,FN)=({TP},{FP},{TN},{FN}); d={d_mu:.3f}")
    print(f"[OUT] {args.out_dir}/rows.csv, rows.json, hist_mu.png, roc_mu.png, mu_vs_s2.png, REPORT.md")

if __name__ == "__main__":
    main()
