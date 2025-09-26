# Efficient WalkSAT with incremental clause counts; then run hierarchical test again with tighter budgets.

import time, json, os, random
from typing import List, Tuple, Sequence
import numpy as np

def parse_dimacs(path: str) -> Tuple[int, int, List[Tuple[int, ...]]]:
    n, m = 0, 0
    clauses: List[Tuple[int, ...]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("c"): continue
            if s.startswith("p"):
                parts = s.split()
                if len(parts) >= 4: n, m = int(parts[-2]), int(parts[-1])
                continue
            xs = [int(x) for x in s.split() if x]
            if xs and xs[-1] == 0: xs.pop()
            if xs: clauses.append(tuple(xs))
    return n, len(clauses), clauses

def chain_rule_closure(n_vars: int, clauses: List[Tuple[int, ...]]):
    from collections import deque
    a = [0] * (n_vars + 1)
    C = len(clauses)
    b = [0] * C
    adj_v_to_c = [[] for _ in range(n_vars + 1)]
    adj_c_to_v = [[] for _ in range(C)]
    for j, cl in enumerate(clauses):
        for lit in cl:
            v = abs(lit); s = +1 if lit > 0 else -1
            adj_c_to_v[j].append((v, s))
            adj_v_to_c[v].append((j, s))
    for start_clause in range(C):
        if b[start_clause] != 0: continue
        b[start_clause] = +1
        dq = deque([("c", start_clause)])
        while dq:
            typ, idx = dq.popleft()
            if typ == "c":
                j = idx; bj = b[j]
                for v, s in adj_c_to_v[j]:
                    need_a = s * bj
                    if a[v] == 0:
                        a[v] = need_a; dq.append(("v", v))
                    elif a[v] != need_a:
                        return False, (a, b)
            else:
                v = idx; av = a[v]
                for j, s in adj_v_to_c[v]:
                    need_b = s * av
                    if b[j] == 0:
                        b[j] = need_b; dq.append(("c", j))
                    elif b[j] != need_b:
                        return False, (a, b)
    return True, (a, b)

# Resonance seed (literal-aware, minimal)
import math
def _next_pow2(x: int) -> int:
    n = 1
    while n < x: n <<= 1
    return n
def _gray(i: int) -> int: return i ^ (i >> 1)
def _walsh_row(N: int, k: int) -> np.ndarray:
    gk = np.uint64(_gray(k))
    n = np.arange(N, dtype=np.uint64)
    bits = n & gk
    u8 = bits.view(np.uint8).reshape(bits.size, 8)
    pc = np.unpackbits(u8, axis=1).sum(axis=1).astype(np.int64)
    return np.where((pc & 1) == 0, 1, -1).astype(np.int8)
def truncated_hadamard(m: int, idx: int = 1) -> np.ndarray:
    if m <= 0: return np.zeros(1, dtype=np.int8)
    N = _next_pow2(m)
    k = idx % N
    if k == 0: k = 1
    return _walsh_row(N, k)[:m]
def stride_near(T: int, frac: float):
    golden = (math.sqrt(5)-1.0)*0.5
    target = int(round((frac % 1.0) * T)) % T
    target = min(max(target, 2), T-2)
    best_s, best_score = 2, 10**9
    for s in range(2, T-1):
        if math.gcd(s, T) != 1: continue
        score = abs(s - target) + 0.04*abs(s - int(round(golden*T)) % T)
        if s in (1,2,3,T-1,T-2,T-3): score += 2.0
        if score < best_score:
            best_s, best_score = s, score
    return best_s
def schedule_literal_aware(n_vars: int, clauses: Sequence[Sequence[int]], seed: int = 42):
    rng = np.random.default_rng(seed)
    C = max(1, len(clauses))
    R = max(1, int(math.ceil(15.0 * math.log(max(2, C)))))
    T = int(3 * R)
    m = max(1, int(math.floor(0.60 * T)))
    sC = stride_near(T, 0.47); sV = stride_near(T, 0.31)
    vote = np.zeros((T, C), dtype=np.int32)
    H_len = _next_pow2(m)
    var_row = {v: (1 + 3 * v) % H_len or 1 for v in range(1, n_vars + 1)}
    var_phase = {v: truncated_hadamard(m, idx=var_row[v]).astype(int) for v in range(1, n_vars + 1)}
    for j, clause in enumerate(clauses):
        offC = (j * sC) % T
        k_neg = int(math.floor(0.40 * m))
        base = truncated_hadamard(m, idx=(17 * j + 11)).astype(int)
        neg_idx = np.where(base < 0)[0]; pos_idx = np.where(base > 0)[0]
        rng.shuffle(neg_idx); rng.shuffle(pos_idx)
        take_neg = set(neg_idx[:min(k_neg, neg_idx.size)])
        extra = k_neg - len(take_neg)
        if extra > 0 and pos_idx.size > 0:
            take_neg |= set(pos_idx[:min(extra, pos_idx.size)])
        for lit in clause:
            v = abs(lit); sign = +1 if (lit > 0) else -1
            row = var_phase[v] * sign
            offV = (sC * v) % T
            for k in range(m):
                tslot = (offC + offV + k) % T
                sgn = -1 if k in take_neg else +1
                vote[tslot, j] += int(sgn * row[k])
    majority = np.sign(np.sum(vote, axis=0, dtype=np.int64)); majority[majority == 0] = 1
    phi = np.ones((T, C), dtype=np.float32) * math.pi
    mask = np.zeros((T, C), dtype=np.uint8)
    for j in range(C):
        col = vote[:, j]
        idx = np.argsort(-np.abs(col))[:m]
        for t in idx:
            mask[t, j] = 1
            phi[t, j] = 0.0 if (np.sign(col[t]) * majority[j] > 0) else math.pi
    return {"phi": phi, "mask": mask, "T": T, "m": m}

def readout_assignment_literal_aware(n_vars: int, clauses: Sequence[Sequence[int]],
                                     phi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    C = len(clauses)
    cosmap = np.zeros(C, dtype=np.float64)
    for j in range(C):
        mcol = mask[:, j].astype(bool)
        if mcol.any():
            cosmap[j] = float(np.mean(np.cos(phi[mcol, j])))
    pos_score = np.zeros(n_vars + 1, dtype=np.float64)
    neg_score = np.zeros(n_vars + 1, dtype=np.float64)
    for j, cl in enumerate(clauses):
        wj = cosmap[j]
        for lit in cl:
            v = abs(lit)
            if lit > 0: pos_score[v] += wj
            else:       neg_score[v] += wj
    bias = pos_score - neg_score
    assign = np.sign(bias[1:])
    assign[assign == 0] = 1
    return assign

# Efficient WalkSAT
def walksat_fast(clauses: Sequence[Sequence[int]], n_vars: int, seed_assign_pm1: np.ndarray,
                 max_flips: int = 120_000, p_random: float = 0.20, restarts: int = 4, rng_seed: int = 42):
    rng = random.Random(rng_seed)

    # Build incidence and literal signed arrays
    clause_lits = []                 # list of (vars, signs) per clause
    pos_occ = [[] for _ in range(n_vars)]
    neg_occ = [[] for _ in range(n_vars)]
    for j, cl in enumerate(clauses):
        vs, ss = [], []
        for lit in cl:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            vs.append(v); ss.append(s)
            (pos_occ if s == 1 else neg_occ)[v].append(j)
        clause_lits.append((vs, ss))

    def init_counts(A_bool):
        num_true = np.zeros(len(clauses), dtype=np.int32)
        for j, (vs, ss) in enumerate(clause_lits):
            t = 0
            for v, s in zip(vs, ss):
                t += 1 if (A_bool[v] if s==1 else (not A_bool[v])) else 0
            num_true[j] = t
        unsat = [j for j in range(len(clauses)) if num_true[j] == 0]
        return num_true, unsat

    def breakcount(v, A_bool, num_true):
        # number of clauses that would become unsatisfied if v flips:
        bc = 0
        if A_bool[v]:  # currently True -> positive literals satisfied; negative ones not
            for j in pos_occ[v]:
                if num_true[j] == 1:
                    bc += 1
        else:          # currently False -> negative literals satisfied; positive ones not
            for j in neg_occ[v]:
                if num_true[j] == 1:
                    bc += 1
        return bc

    def flip_var(v, A_bool, num_true):
        # Update num_true incrementally
        if A_bool[v]:
            # True -> False: positive literals lose satisfaction; negative gain
            for j in pos_occ[v]:
                num_true[j] -= 1
            for j in neg_occ[v]:
                num_true[j] += 1
        else:
            # False -> True
            for j in pos_occ[v]:
                num_true[j] += 1
            for j in neg_occ[v]:
                num_true[j] -= 1
        A_bool[v] = not A_bool[v]

    best_model = None
    for restart in range(restarts):
        A_bool = (seed_assign_pm1.copy() == +1)
        # random perturbation per restart
        for _ in range(restart * 3):
            i = rng.randrange(n_vars); A_bool[i] = not A_bool[i]
        num_true, unsat = init_counts(A_bool)
        if not unsat:
            return np.where(A_bool, +1, -1)
        for flips in range(max_flips):
            cj = rng.choice(unsat)
            vs, ss = clause_lits[cj]
            if rng.random() < p_random:
                v = rng.choice(vs)
            else:
                # choose var with minimal breakcount; break ties randomly
                bcs = [(breakcount(v, A_bool, num_true), v) for v in vs]
                min_bc = min(bcs)[0]
                cand = [v for bc, v in bcs if bc == min_bc]
                v = rng.choice(cand)
            # flip
            flip_var(v, A_bool, num_true)
            # update unsat: only clauses affected by v can change status
            # Recompute unsat list efficiently: track changes
            # We'll rebuild unsat cheaply by examining affected clauses plus remove satisfied ones
            # Cheap approach: rebuild full unsat every k flips
            if flips % 200 == 0:
                unsat = [j for j in range(len(clauses)) if num_true[j] == 0]
            else:
                # incremental update: check only affected clauses
                affected = set(pos_occ[v] + neg_occ[v])
                # ensure uniqueness
                unsat_set = set(unsat)
                for j in affected:
                    if num_true[j] == 0:
                        unsat_set.add(j)
                    elif j in unsat_set:
                        unsat_set.remove(j)
                unsat = list(unsat_set)
            if not unsat:
                return np.where(A_bool, +1, -1)
                #return np
    return best_model

def resonance_seed_then_walksat(path: str, seed=42):
    n, m, clauses = parse_dimacs(path)
    sched = schedule_literal_aware(n, clauses, seed=seed)
    phi, mask = sched["phi"], sched["mask"]
    assign_seed = readout_assignment_literal_aware(n, clauses, phi, mask)
    sat_ratio_seed = float(np.mean([any(assign_seed[abs(lit)-1] == (1 if lit>0 else -1) for lit in cl) for cl in clauses]))
    model = walksat_fast(clauses, n, assign_seed, max_flips=90000, p_random=0.20, restarts=4, rng_seed=seed)
    return n, m, sat_ratio_seed, model

def hierarchical_test_fast(path: str, seed=42):
    n, m, clauses = parse_dimacs(path)
    t0 = time.time()
    ok, final_model = chain_rule_closure(n, clauses)
    if ok:
        verdict = "SANITY"; stage = "chain-closure"; model_found = True
        sat_ratio_seed = 1.0
    else:
        n, m, sat_ratio_seed, model = resonance_seed_then_walksat(path, seed=seed)
        model_found = (model is not None)
        verdict = "SANITY" if model_found else "INSANITY"
        stage = "walksat" if model_found else "fail"
    dt = time.time() - t0
    return dict(file=path, verdict=verdict, stage=stage, sat_ratio_seed=round(sat_ratio_seed,6),
                model_found=bool(model_found), n_vars=n, n_clauses=m, elapsed_sec=dt, final_model=final_model)

# Run both now with the faster WalkSAT
paths = ["sample.cnf", "uf250-098.cnf", "uf250-099.cnf", "uf250-0100.cnf", "uuf250-098.cnf", "uuf250-099.cnf", "uuf250-0100.cnf"]
reports = []
for p in paths:
    reports.append(hierarchical_test_fast(p, seed=42))

# Save reports
for r in reports:
    with open(f"/mnt/data/{os.path.basename(r['file'])}.hier_fast_report.json", "w") as f:
        json.dump(r, f, indent=2)

    print(r)
