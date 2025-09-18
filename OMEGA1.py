#!/usr/bin/env python3
# AO_SUITE.py - Hypercube closure probe + CNF S2 + simple arbiter + holonomy + rational windows
# Requires: numpy

import argparse, math, numpy as np
from numpy.linalg import eigh

# ------------------ Utilities ------------------

def angles_mod(x):
    return (x + np.pi) % (2*np.pi) - np.pi

# ------------------ DIMACS CNF ------------------

def parse_dimacs(path):
    n_vars = n_clauses = None
    clauses = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                parts = line.split()
                if len(parts) >= 4:
                    n_vars = int(parts[2]); n_clauses = int(parts[3])
                continue
            lits = [int(x) for x in line.split() if x]
            if lits and lits[-1] == 0: lits = lits[:-1]
            if lits: clauses.append(lits)
    if n_vars is None or n_clauses is None:
        n_clauses = len(clauses)
        n_vars = max(abs(x) for cl in clauses for x in cl) if clauses else 0
    return n_vars, n_clauses, clauses

def clause_satisfied(clause, assign):
    # assign: dict var->bool
    for lit in clause:
        v = abs(lit); sign = (lit > 0)
        if assign.get(v, False) == sign:
            return True
    return False

def majority_assignment(n_vars, clauses):
    # majority of literal polarity per var; ties -> False
    pos = np.zeros(n_vars+1, dtype=int)
    neg = np.zeros(n_vars+1, dtype=int)
    for cl in clauses:
        for lit in cl:
            if lit > 0: pos[abs(lit)] += 1
            else:       neg[abs(lit)] += 1
    assign = {}
    for v in range(1, n_vars+1):
        if pos[v] > neg[v]: assign[v] = True
        elif pos[v] < neg[v]: assign[v] = False
        else: assign[v] = False
    return assign

# ------------------ Hypercube Face Closure ------------------

def face_closure_stats(n_vars, clauses, sample_pairs=512, eps=0.10, seed=0):
    rng = np.random.default_rng(seed)
    base = majority_assignment(n_vars, clauses)

    # Precompute per-clause base φ in {0,π}
    def phi_for(assign):
        # total phase = sum over clauses of 0 or π
        val = 0.0
        for cl in clauses:
            val += (0.0 if clause_satisfied(cl, assign) else np.pi)
        return angles_mod(val)

    # sample variable pairs
    if sample_pairs > n_vars*(n_vars-1)//2:
        sample_pairs = n_vars*(n_vars-1)//2
    pairs = set()
    while len(pairs) < sample_pairs:
        i = rng.integers(1, n_vars+1)
        j = rng.integers(1, n_vars+1)
        if i == j: continue
        a,b = (i,j) if i<j else (j,i)
        pairs.add((a,b))
    pairs = list(pairs)

    d0_vals = []
    for (i,j) in pairs:
        # four corners of the 2D face (flip i and j around base)
        a00 = base.copy()
        a10 = base.copy(); a10[i] = not a10[i]
        a01 = base.copy(); a01[j] = not a01[j]
        a11 = base.copy(); a11[i] = not a11[i]; a11[j] = not a11[j]

        phi00 = phi_for(a00)
        phi10 = phi_for(a10)
        phi01 = phi_for(a01)
        phi11 = phi_for(a11)

        # closure sum on the face
        dphi = angles_mod(phi00 + phi10 + phi01 + phi11)
        # distance to nearest multiple of 2π
        d0 = float(abs(angles_mod(dphi)))
        d0_vals.append(d0)

    d0_vals = np.array(d0_vals)
    med = float(np.median(d0_vals))
    mean = float(np.mean(d0_vals))
    frac_ok = float(np.mean(d0_vals <= eps))
    return dict(median_d0=med, mean_d0=mean, frac_ok=frac_ok, eps=eps, samples=len(d0_vals))

# ------------------ AO-style UNSAT-Hadamard & S2 on CNF graph ------------------

def neighbors_from_cnf(n_clauses, clauses):
    var_to_clauses = {}
    for ci, cl in enumerate(clauses):
        for lit in cl:
            var_to_clauses.setdefault(abs(lit), []).append(ci)
    nbrs = [set() for _ in range(n_clauses)]
    for idxs in var_to_clauses.values():
        L = len(idxs)
        for a in range(L):
            ca = idxs[a]
            for b in range(a+1, L):
                cb = idxs[b]
                nbrs[ca].add(cb); nbrs[cb].add(ca)
    degs = np.array([len(s) for s in nbrs], dtype=int)
    edges = [(i,j) for i in range(n_clauses) for j in nbrs[i] if j>i]
    return nbrs, degs, edges

def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H],[H, -H]])
    return H

def next_pow2(n):
    k = 1
    while k < n: k <<= 1
    return k

def stride_near_half_coprime(T):
    s = max(1, T//2 - 1)
    while math.gcd(s, T) != 1:
        s -= 1
        if s <= 0:
            s = 1
            break
    return s

def schedule_unsat_hadamard(C, R, rho_lock=0.5, zeta0=0.4, L=3, seed=42, return_locks=True):
    rng = np.random.default_rng(seed)
    T = R*L; m = int(round(rho_lock*T)); k = int(round(zeta0*m))
    s = stride_near_half_coprime(T)
    offsets = [(j*s) % T for j in range(C)]
    locks = [np.array([(offsets[j]+t)%T, dtype=int) for t in range(m)] for j in range(C)]

    Phi = np.full((T,C), np.pi, float)
    Hlen = next_pow2(m); H = hadamard(Hlen)
    row_step = (Hlen//2) + 1
    while math.gcd(row_step, Hlen) != 1: row_step += 2
    g = (Hlen//3) | 1
    while math.gcd(g, Hlen) != 1: g += 2
    cols = (g * np.arange(m)) % Hlen

    for j in range(C):
        row = H[(j*row_step)%Hlen, cols]
        neg = np.flatnonzero(row < 0.0)
        if len(neg) >= k:
            mask_pi = rng.choice(neg, size=k, replace=False)
        else:
            extra = rng.choice(np.setdiff1d(np.arange(m), neg), size=k-len(neg), replace=False)
            mask_pi = np.concatenate([neg, extra])
        mask_0 = np.setdiff1d(np.arange(m), mask_pi)
        slots = locks[j]
        Phi[slots[mask_pi], j] = np.pi
        Phi[slots[mask_0], j]  = 0.0
    return (Phi, locks) if return_locks else Phi

def gram_complex(Phi):
    Z = np.exp(1j*Phi)
    G = (Z.conj().T @ Z) / Phi.shape[0]
    G = 0.5*(G + G.conj().T)
    np.fill_diagonal(G, 1.0+0j)
    return G

def top_mu_lambda_power(Z, iters=50, tol=1e-9):
    T, C = Z.shape
    rng = np.random.default_rng(0)
    v = rng.standard_normal(C) + 1j*rng.standard_normal(C)
    v = v/np.linalg.norm(v)
    last = 0.0
    for _ in range(iters):
        u = Z @ v
        w = (Z.conj().T @ u) / T
        lam = float(np.vdot(v, w).real)
        nrm = np.linalg.norm(w)
        if nrm == 0: break
        v = w/nrm
        if abs(lam - last) < tol*max(1.0, abs(lam)): break
        last = lam
    return lam/C, lam

def s2_metrics_edges(Z, edges, lock_masks=None, m=None):
    T, C = Z.shape
    row_sum = np.zeros(C, dtype=float)
    abs_vals = []
    conjZ = Z.conj()
    for i,j in edges:
        if lock_masks is None:
            val = (conjZ[:, i] * Z[:, j]).sum() / T
        else:
            inter = lock_masks[:, i] & lock_masks[:, j]
            if inter.any():
                val = (conjZ[inter, i] * Z[inter, j]).sum() / m
            else:
                val = 0.0
        a = float(abs(val))
        abs_vals.append(a)
        row_sum[i] += a; row_sum[j] += a
    abs_vals = np.array(abs_vals) if abs_vals else np.array([0.0])
    return dict(avg_edge=float(abs_vals.mean()), max_edge=float(abs_vals.max()),
                avg_row_sum=float(row_sum.mean()), max_row_sum=float(row_sum.max()))

def kappa_S2(m, T, zeta0):
    term1 = (1.0 - 2.0*zeta0)**2
    pow2 = 1 << int(np.ceil(np.log2(max(1, m))))
    term2 = pow2**-0.5
    term3 = 2.0/m
    term4 = 1.0/T
    return term1 + term2 + term3 + term4

# ------------------ Holonomy boundary demo ------------------

def holonomy_boundary(phi_grid):
    Sx, Sy = phi_grid.shape
    dfx = angles_mod(np.diff(phi_grid, axis=0, append=phi_grid[:1, :]))
    dfy = angles_mod(np.diff(phi_grid, axis=1, append=phi_grid[:, :1]))
    curl = angles_mod(dfx[:-1, :-1] + dfy[1:, :-1] - dfx[:-1, 1:] - dfy[:-1, :-1])
    vals = np.abs(curl).ravel()
    med = float(np.median(vals)); mean = float(np.mean(vals))
    return dict(median_curl=med, mean_curl=mean)

def make_boundary_grid(S=16, kappa=0.10, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2*np.pi, S, endpoint=False)
    y = np.linspace(0, 2*np.pi, S, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    phi = 0.2*np.sin(X) + 0.2*np.cos(Y) + kappa*np.sin(X+Y)
    phi += 0.03 * rng.standard_normal(phi.shape)
    return angles_mod(phi)

# ------------------ Rational pi windows (logistic map) ------------------

def logistic_windows(r=3.8, eps=0.02, N=200000, burn=5000, qs=32, seed=0):
    rng = np.random.default_rng(seed)
    def evolve(omega):
        x = 0.123456789
        mu_acc = 0.0; lam_acc = 0.0
        for t in range(burn+N):
            x = min(1.0, max(0.0, r*x*(1-x) + eps*math.sin(omega*t)))
            if t >= burn:
                mu_acc += x
                lam_acc += math.log(abs(r*(1-2*x)) + 1e-12)
        mu = mu_acc/N
        lam = lam_acc/N
        return mu, lam
    out = []
    for q in range(1, qs+1):
        for p in range(1, q+1):
            omega = math.pi * p/q
            mu, lam = evolve(omega)
            out.append(dict(p=p, q=q, omega=omega, mu=mu, lam=lam))
    # add irrational reference pi*sqrt(2)
    omega = math.pi*math.sqrt(2.0); mu, lam = evolve(omega)
    out.append(dict(p='irr', q='sqrt2', omega=omega, mu=mu, lam=lam))
    return out

# ------------------ Arbiter ------------------

def decide_from_cnf(path, sample_pairs=512, eps_face=0.10, cR=15.0, rho=0.5, zeta0=0.4, seed=42):
    n_vars, n_clauses, clauses = parse_dimacs(path)
    # Hypercube face test
    hc = face_closure_stats(n_vars, clauses, sample_pairs=sample_pairs, eps=eps_face, seed=seed)
    # S2 on actual neighbor graph with UNSAT schedule
    nbrs, degs, edges = neighbors_from_cnf(n_clauses, clauses)
    C = n_clauses; R = max(1, int(math.ceil(cR * math.log(max(2, C))))); T = 3*R
    Phi, locks = schedule_unsat_hadamard(C, R, rho, zeta0, L=3, seed=seed, return_locks=True)
    Z = np.exp(1j*Phi)
    lock_mask = np.zeros((T, C), dtype=bool)
    for j in range(C): lock_mask[locks[j], j] = True
    s2_lock = s2_metrics_edges(Z, edges, lock_masks=lock_mask, m=len(locks[0]))
    kappa = kappa_S2(len(locks[0]), T, zeta0)
    bound_avg = float(degs.mean()) * kappa

    # Votes
    votes = []
    votes.append(('hypercube', hc['median_d0'] <= eps_face))
    votes.append(('S2_lock', s2_lock['avg_row_sum'] <= bound_avg))

    sat_votes = sum(v for _, v in votes)
    decision = 'SAT' if sat_votes >= 2 else 'UNSAT'

    return dict(
        file=str(Path(path).name), vars=n_vars, clauses=n_clauses,
        hypercube=hc, S2_lock=s2_lock, deg_avg=float(degs.mean()), kappa_S2=kappa,
        bound_avg_d=bound_avg, votes=votes, decision=decision
    )

# ------------------ CLI ------------------

def main():
    ap = argparse.ArgumentParser(description="AO Suite: hypercube, S2, holonomy, rational windows")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("hypercube")
    p1.add_argument("cnf", type=str)
    p1.add_argument("--pairs", type=int, default=512)
    p1.add_argument("--eps", type=float, default=0.10)
    p1.add_argument("--seed", type=int, default=0)

    p2 = sub.add_parser("s2")
    p2.add_argument("cnf", type=str)
    p2.add_argument("--cR", type=float, default=15.0)
    p2.add_argument("--rho", type=float, default=0.5)
    p2.add_argument("--zeta0", type=float, default=0.4)
    p2.add_argument("--seed", type=int, default=42)

    p3 = sub.add_parser("decide")
    p3.add_argument("cnf", type=str)
    p3.add_argument("--pairs", type=int, default=512)
    p3.add_argument("--eps", type=float, default=0.10)
    p3.add_argument("--cR", type=float, default=15.0)
    p3.add_argument("--rho", type=float, default=0.5)
    p3.add_argument("--zeta0", type=float, default=0.4)
    p3.add_argument("--seed", type=int, default=42)

    p4 = sub.add_parser("holonomy_demo")
    p4.add_argument("--S", type=int, default=16)
    p4.add_argument("--kappa", type=float, default=0.10)
    p4.add_argument("--seed", type=int, default=0)

    p5 = sub.add_parser("rational_windows")
    p5.add_argument("--r", type=float, default=3.8)
    p5.add_argument("--eps", type=float, default=0.02)
    p5.add_argument("--N", type=int, default=100000)
    p5.add_argument("--burn", type=int, default=5000)
    p5.add_argument("--qs", type=int, default=16)
    p5.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if args.cmd == "hypercube":
        n_vars, n_clauses, clauses = parse_dimacs(args.cnf)
        out = face_closure_stats(n_vars, clauses, sample_pairs=args.pairs, eps=args.eps, seed=args.seed)
        print(json.dumps(out, indent=2))
    elif args.cmd == "s2":
        n_vars, n_clauses, clauses = parse_dimacs(args.cnf)
        nbrs, degs, edges = neighbors_from_cnf(n_clauses, clauses)
        C = n_clauses; R = max(1, int(math.ceil(args.cR * math.log(max(2, C))))); T = 3*R
        Phi, locks = schedule_unsat_hadamard(C, R, args.rho, args.zeta0, L=3, seed=args.seed, return_locks=True)
        Z = np.exp(1j*Phi)
        lock_mask = np.zeros((T, C), dtype=bool)
        for j in range(C): lock_mask[locks[j], j] = True
        s2_all = s2_metrics_edges(Z, edges, lock_masks=None)
        s2_lock = s2_metrics_edges(Z, edges, lock_masks=lock_mask, m=len(locks[0]))
        kappa = kappa_S2(len(locks[0]), T, args.zeta0)
        out = dict(clauses=n_clauses, R=R, T=T, m=len(locks[0]),
                   deg_avg=float(degs.mean()), deg_max=int(degs.max()),
                   s2_all=s2_all, s2_lock=s2_lock,
                   kappa_S2=kappa, bound_avg_d=kappa*float(degs.mean()))
        print(json.dumps(out, indent=2))
    elif args.cmd == "decide":
        out = decide_from_cnf(args.cnf, sample_pairs=args.pairs, eps_face=args.eps,
                              cR=args.cR, rho=args.rho, zeta0=args.zeta0, seed=args.seed)
        print(json.dumps(out, indent=2))
    elif args.cmd == "holonomy_demo":
        grid = make_boundary_grid(S=args.S, kappa=args.kappa, seed=args.seed)
        stats = holonomy_boundary(grid)
        print(json.dumps(dict(S=args.S, kappa=args.kappa, stats=stats), indent=2))
    elif args.cmd == "rational_windows":
        rows = logistic_windows(r=args.r, eps=args.eps, N=args.N, burn=args.burn, qs=args.qs, seed=args.seed)
        # print a compact CSV-ish table
        for row in rows[:10]:
            print(f"p={row['p']}, q={row['q']}, mu={row['mu']:.4f}, lam={row['lam']:.4f}")
        print(f"... total results: {len(rows)}")
    else:
        raise SystemExit(2)

if __name__ == "__main__":
    main()

#path = Path("/mnt/data/AO_SUITE.py")
#path.write_text(code, encoding="utf-8")
#print("Wrote", path)

# Quick smoke test on one of the small u(f/uuf) files with limited pairs to keep runtime reasonable here.
#test_cnf = "/mnt/data/uf250-098.cnf"
#if Path(test_cnf).exists():
#    cmd = f'python "{path}" decide "{test_cnf}" --pairs 128 --eps 0.10 --seed 0'
#    res = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=120)
#    print("DECIDE uf250-098 (128 pairs):\n", res.stdout[:800])
#else:
#    print("CNF not present for test run.")
