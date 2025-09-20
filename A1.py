#!/usr/bin/env python3
import argparse, json, math, sys
import numpy as np

# -------------------- Deterministic helpers --------------------

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
    if n_vars is None:
        # Fallback if header missing
        n_vars = max(abs(l) for cls in clauses for l in cls)
    return n_vars, clauses

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

def coprime_stride(n, seed=1):
    # choose odd stride near n//2 that is coprime with n
    # deterministic scan around n//2
    def gcd(a,b):
        while b: a,b=b,a%b
        return a
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
    # stride near T/2, coprime with T
    s = coprime_stride(T, seed=1)
    offs = [(j*s) % T for j in range(C)]
    return offs, s

def var_row(i, H_len):
    # deterministic odd stride mapping of variables to hadamard rows
    return (1 + 3*i) % H_len  # 3 is odd; H_len is power of 2

# -------------------- CNF → Phi (core) --------------------

def build_phi_from_cnf(T, m, clauses, zeta0, col_stride=1, seed=1):
    """
    T = total slots, m = lock length, clauses = list of lists of signed ints
    """
    C = len(clauses)
    Phi = np.full((T, C), np.pi, dtype=np.float64)

    # Hadamard length is next pow2 of m
    H_len = next_pow2(m)
    H = sylvester_hadamard(H_len)

    # De-aliased clause offsets
    offsets, stride_T = dealiased_offsets(C, T)

    # Deterministic column walk
    col_stride = col_stride if col_stride % 2 == 1 else (col_stride+1)
    sub_idx = [(k*col_stride) % H_len for k in range(m)]

    kk = int(max(0, min(m, round(zeta0*m))))  # how many π-slots per clause

    for j, lits in enumerate(clauses):
        agg = np.zeros(m, dtype=np.int32)
        # accumulate literal-sign rows
        for lit in lits:
            v = abs(lit)
            r = var_row(v, H_len)
            row = H[r, sub_idx].copy()  # ±1
            if lit < 0:
                row *= -1
            agg += row
        # choose kk most-negative to set π, rest 0
        order = np.argsort(agg)  # most negative first
        neg_set = set(order[:kk])
        start = offsets[j]
        for k in range(m):
            t = (start + k) % T
            Phi[t, j] = np.pi if k in neg_set else 0.0
    return Phi

# -------------------- Spectral tester --------------------

def spectral_mu(Phi):
    Z = np.exp(1j*Phi)       # shape T x C
    T = Phi.shape[0]
    G = (Z.conj().T @ Z) / T # C x C Hermitian
    # power method (deterministic init)
    C = G.shape[0]
    v = np.ones((C,), dtype=complex)
    v /= np.linalg.norm(v)
    for _ in range(80):
        v = G @ v
        nv = np.linalg.norm(v)
        if nv == 0: break
        v /= nv
    lam = float(np.real(np.vdot(v, G @ v)))
    mu = lam / C
    return mu, lam, G

def s2_metrics(G, edges=None, lock_mask=None):
    """
    Basic S2-style summaries:
      - |G_ij| over edges: max/avg
      - row-sum over neighbors: max/avg
    If edges is None, use all off-diagonals as neighbors (dense).
    """
    C = G.shape[0]
    A = np.abs(G)
    if edges is None:
        # full graph (exclude diagonal)
        mask = np.ones_like(A, dtype=bool)
        np.fill_diagonal(mask, 0)
        vals = A[mask]
        max_abs = float(np.max(vals)) if vals.size else 0.0
        avg_abs = float(np.mean(vals)) if vals.size else 0.0
        row_sums = (A - np.diag(np.diag(A))).sum(axis=1)
        return {
            "abs_edge_max": max_abs,
            "abs_edge_avg": avg_abs,
            "row_sum_max": float(np.max(row_sums)),
            "row_sum_avg": float(np.mean(row_sums))
        }
    else:
        # custom edges (list of (i,j))
        vals = [A[i,j] for (i,j) in edges]
        max_abs = float(np.max(vals)) if len(vals)>0 else 0.0
        avg_abs = float(np.mean(vals)) if len(vals)>0 else 0.0
        # Build neighbor row sums
        deg = [0.0]*C
        for (i,j) in edges:
            deg[i] += A[i,j]
            deg[j] += A[j,i]
        return {
            "abs_edge_max": max_abs,
            "abs_edge_avg": avg_abs,
            "row_sum_max": float(np.max(deg)) if deg else 0.0,
            "row_sum_avg": float(np.mean(deg)) if deg else 0.0
        }

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnf", required=True, help="DIMACS CNF path")
    ap.add_argument("--cR", type=float, default=15.0, help="R ≈ cR * log C")
    ap.add_argument("--rho_lock", type=float, default=0.50)
    ap.add_argument("--zeta0", type=float, default=0.40)
    ap.add_argument("--col_stride", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau", type=float, default=None, help="Decision threshold for mu")
    ap.add_argument("--report_json", default=None)
    args = ap.parse_args()

    n_vars, clauses = parse_dimacs(args.cnf)
    C = len(clauses)
    if C == 0:
        print("Empty CNF.", file=sys.stderr)
        sys.exit(2)

    R = max(3, int(round(args.cR * math.log(max(C,2)))))
    T = 3*R
    m = int(round(args.rho_lock * T))

    np.random.seed(args.seed)

    Phi = build_phi_from_cnf(T=T, m=m, clauses=clauses,
                             zeta0=args.zeta0, col_stride=args.col_stride)

    mu, lam, G = spectral_mu(Phi)
    s2 = s2_metrics(G)

    decision = None
    if args.tau is not None:
        decision = "SAT" if mu >= args.tau else "UNSAT"

    out = {
        "cnf": args.cnf,
        "n_vars": n_vars,
        "C": C, "R": R, "T": T, "m": m,
        "rho_lock": args.rho_lock, "zeta0": args.zeta0,
        "col_stride": args.col_stride,
        "mu": mu, "lambda_max": lam,
        "s2": s2,
        "tau": args.tau,
        "decision": decision
    }

    print(f"C={C}, R={R}, T={T}, m={m}")
    print(f"mu={mu:.6f}, lambda_max={lam:.3f}")
    print("S2:", s2)
    if decision is not None:
        print("Decision:", decision)

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
