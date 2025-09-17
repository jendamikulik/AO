#!/usr/bin/env python3
# resobench.py — Practical resonance benchmark pack
# - CNF parse/generate (random3, pigeonhole PHP(p,p-1))
# - Resonance margin scan (median dist-to-0 vs dist-to-π/2 on probe grid)
# - Small-n brute SAT truth
# - Batch runner -> CSV

import argparse, math, os, sys, csv, itertools, random
from pathlib import Path
import numpy as np

# ---------------- DIMACS ----------------
def parse_dimacs(path):
    n = 0
    clauses = []
    cur = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('c'):  # comments
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
    return n, clauses

def write_dimacs(path, n, clauses, comment=None):
    with open(path, "w", encoding="utf-8") as f:
        if comment:
            for line in comment.splitlines():
                f.write(f"c {line}\n")
        f.write(f"p cnf {n} {len(clauses)}\n")
        for cl in clauses:
            f.write(" ".join(str(x) for x in cl) + " 0\n")

# --------------- Generators ---------------
def gen_random_3sat(n, m, seed=0):
    rng = random.Random(seed)
    clauses = []
    for _ in range(m):
        vars3 = rng.sample(range(1, n+1), k=3)
        lits = []
        for v in vars3:
            sign = 1 if rng.random() < 0.5 else -1
            lits.append(sign * v)
        clauses.append(tuple(lits))
    return n, clauses

def gen_php(p, q=None):
    """
    Pigeonhole principle in CNF:
      q = p-1 -> UNSAT
      variables x_{i,j} : pigeon i in hole j   (i=1..p, j=1..q)
    Clauses:
      (A) Each pigeon in at least one hole: (x_{i,1} ∨ ... ∨ x_{i,q})
      (B) No hole contains two pigeons: for each j and i<k: (¬x_{i,j} ∨ ¬x_{k,j})
    """
    if q is None:
        q = p-1
    # map (i,j) -> var id
    def vid(i,j): return (i-1)*q + j
    n = p*q
    clauses = []
    # (A)
    for i in range(1, p+1):
        clauses.append(tuple(vid(i,j) for j in range(1, q+1)))
    # (B)
    for j in range(1, q+1):
        for i in range(1, p+1):
            for k in range(i+1, p+1):
                clauses.append((-vid(i,j), -vid(k,j)))
    return n, clauses

# --------------- Truth for small n ---------------
def eval_clause(clause, assign_bits):
    # assign_bits: dict var->bool
    for lit in clause:
        v = abs(lit)
        val = assign_bits[v]
        if lit < 0:
            val = (not val)
        if val:
            return True
    return False

def is_sat_bruteforce(n, clauses, max_n_truth=22):
    if n > max_n_truth:
        return None  # skip (too big)
    for bits in range(1<<n):
        a = {i+1: bool((bits>>i)&1) for i in range(n)}
        if all(eval_clause(cl, a) for cl in clauses):
            return True
    return False

# --------------- Resonance core ---------------
def count_unsat(clauses, sigma_pm):
    """sigma_pm: np.array shape (n,), entries in {+1,-1} meaning True/False."""
    u = 0
    for cl in clauses:
        sat = False
        for lit in cl:
            j = abs(lit) - 1
            sign = sigma_pm[j]  # +1 for True, -1 for False
            val = (sign == +1) if (lit > 0) else (sign == -1)
            if val:
                sat = True
                break
        if not sat:
            u += 1
    return u

def delta_phi_from_u(u, x, t, omega):
    """
    Minimal π-rational model:
      phase = π*x + ω t + 2π * ( (u / 4) mod 1 )
      then wrap to (-π, π]
    """
    total = math.pi*x + omega*t + 2.0*math.pi*((u/4.0) % 1.0)
    n_star = int(round(total / (2.0*math.pi)))
    dphi = ((total - 2.0*math.pi*n_star + math.pi) % (2.0*math.pi)) - math.pi
    return dphi

def dist_zero_mod_pi(a):
    """Shortest distance to 0 on circle with period π."""
    x = abs(a) % math.pi
    return min(x, math.pi - x)

def dist_half_mod_pi(a):
    """Distance to π/2 on circle with period π."""
    x = abs(a) % math.pi
    return abs(x - math.pi/2.0)

def resonance_margin(clauses, n,
                     use_dual_x=True, dx=0.01, unit_jitter=0.02,
                     use_sigma_sym=True):
    """
    Grid over probes:
      x ∈ {0.25,0.75} ± dx
      σ ∈ {(+1)^n, (-1)^n} (optional symmetry)
      "unit jitter" realized as perturbation of the u→phase map via small offsets:
        add ±ε to u/4 before mod 1 (numerically same as small phase shift)
    Decision:
      med_d0 vs med_dhalf over the grid.
    """
    t, omega = 1.0, 2.0*math.pi

    xs_main = (0.25, 0.75) if use_dual_x else (0.25,)
    xs = []
    for xm in xs_main:
        xs.extend([xm - dx, xm, xm + dx])

    sigmas = [np.ones(n, dtype=int)]
    if use_sigma_sym:
        sigmas.append(-np.ones(n, dtype=int))

    unit_eps = [0.0, -unit_jitter, +unit_jitter]

    d0s, dHs = [], []
    strict0 = strictH = 0
    best = None  # (min(d0,dH), dphi, x, sign)

    for x in xs:
        for sigma in sigmas:
            sign = int(np.sign(sigma[0]) or 1)
            u = count_unsat(clauses, sigma)
            for eps in unit_eps:
                # jitter: u/4 -> u/4 + eps
                total = math.pi*x + omega*t + 2.0*math.pi*(((u/4.0)+eps) % 1.0)
                n_star = int(round(total / (2.0*math.pi)))
                dphi = ((total - 2.0*math.pi*n_star + math.pi) % (2.0*math.pi)) - math.pi
                d0  = dist_zero_mod_pi(dphi)
                dH  = dist_half_mod_pi(dphi)
                d0s.append(d0); dHs.append(dH)
                if d0 < 1e-2: strict0 += 1
                if dH < 1e-2: strictH += 1
                cand = (min(d0,dH), dphi, x, sign)
                if best is None or cand[0] < best[0]:
                    best = cand

    med0 = float(np.median(d0s)) if d0s else float("inf")
    medH = float(np.median(dHs)) if dHs else float("inf")
    margin = med0 - medH   # >0 ⇒ ZERO closer ⇒ SAT ; <0 ⇒ HALF closer ⇒ UNSAT
    guess = "SAT" if margin > 0 else "UNSAT" if margin < 0 else "UNKNOWN"
    rule  = "median-zero-distance" if margin > 0 else "median-half-distance" if margin < 0 else "median-tie"

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
    return guess, info

# --------------- Batch & CSV ---------------
def run_files(paths, brute_truth=True, max_truth_n=22, seed=0, out_csv=None):
    rows = []
    for p in paths:
        n, cls = parse_dimacs(p)
        guess, info = resonance_margin(cls, n)

        truth = is_sat_bruteforce(n, cls, max_n_truth=max_truth_n) if brute_truth else None
        rows.append({
            "file": Path(p).name,
            "n": n,
            "m": len(cls),
            "guess": guess,
            "margin": f"{info['margin']:.6f}",
            "med_d0": f"{info['med_d0']:.6f}",
            "med_dhalf": f"{info['med_dhalf']:.6f}",
            "rule": info["rule"],
            "strict0": info["strict0"],
            "strictHalf": info["strictHalf"],
            "truth": ("SAT" if truth else "UNSAT") if truth is not None else "",
        })
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)
    return rows

def find_cnf_in_dir(d):
    return sorted(str(p) for p in Path(d).glob("*.cnf"))

# --------------- CLI ---------------
def main():
    ap = argparse.ArgumentParser(description="Resonance benchmark (generate, scan, CSV)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g1 = sub.add_parser("gen3sat", help="Generate random 3-SAT")
    g1.add_argument("-n", type=int, required=True)
    g1.add_argument("-m", type=int, required=True)
    g1.add_argument("-o", "--out", required=True)
    g1.add_argument("--seed", type=int, default=0)

    g2 = sub.add_parser("genphp", help="Generate PHP(p,p-1) UNSAT (or p,q)")
    g2.add_argument("-p", type=int, required=True)
    g2.add_argument("-q", type=int, default=None)
    g2.add_argument("-o", "--out", required=True)

    s1 = sub.add_parser("scan", help="Scan files or a directory")
    s1.add_argument("paths", nargs="+", help="CNF files or directories")
    s1.add_argument("--csv", help="Write results to CSV")
    s1.add_argument("--no_truth", action="store_true", help="Disable brute SAT truth (faster)")

    args = ap.parse_args()

    if args.cmd == "gen3sat":
        n, cls = gen_random_3sat(args.n, args.m, seed=args.seed)
        write_dimacs(args.out, n, cls, comment=f"random3 n={args.n} m={args.m} seed={args.seed}")
        print(f"Wrote {args.out} (n={n}, m={len(cls)})")

    elif args.cmd == "genphp":
        n, cls = gen_php(args.p, args.q)
        comment = f"PHP({args.p},{args.q if args.q else args.p-1})"
        write_dimacs(args.out, n, cls, comment=comment)
        print(f"Wrote {args.out} (n={n}, m={len(cls)})")

    elif args.cmd == "scan":
        files = []
        for p in args.paths:
            if os.path.isdir(p):
                files.extend(find_cnf_in_dir(p))
            else:
                files.append(p)
        if not files:
            print("No CNF files found.", file=sys.stderr)
            sys.exit(1)
        rows = run_files(files, brute_truth=(not args.no_truth), out_csv=args.csv)
        for r in rows:
            print(r)

if __name__ == "__main__":
    main()
