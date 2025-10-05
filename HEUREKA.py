import math, random
import numpy as np
#import WINNER_ALL_v3 as W

HBAR = 1.054_571_817e-34
H = 2.0 * math.pi * HBAR
Q_E = 1.602_176_634e-19  # electron charge

def parse_dimacs_cnf(path):
    clauses = []
    n_vars = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            cur = []
            for line in f:
                line = line.strip()
                if not line or line.startswith('c'):
                    continue
                if line.startswith('p'):
                    parts = line.split()
                    if len(parts) >= 4 and parts[1].lower() == 'cnf':
                        n_vars = int(parts[2])
                    continue
                for tok in line.split():
                    lit = int(tok)
                    if lit == 0:
                        if cur:
                            clauses.append(tuple(cur))
                            cur = []
                    else:
                        cur.append(lit)
            if cur:
                clauses.append(tuple(cur))
        if n_vars is None and clauses:
            n_vars = max(abs(x) for cl in clauses for x in cl)
        elif n_vars is None:
            n_vars = 0
        return n_vars, clauses
    except FileNotFoundError:
        print(f"Soubor {path} nenalezen.")
        return 0, []
    except ValueError:
        print(f"Chyba ve formátu souboru {path}: neplatný literál.")
        return 0, []
    except Exception as e:
        print(f"Chyba při parsování {path}: {str(e)}")
        return 0, []

def dimacs_to_clause(lits):
    return tuple((abs(lit) - 1, lit > 0) for lit in lits)  # -1 protože indexy začínají od 0

# CNF klauzule jako [(var_idx, is_positive), (..), (..)]
# u_c(sigma) = 1 (false), jinak 0
def clause_unsat(clause, sigma):
    sat = False
    for (i, is_pos) in clause:
        lit = (sigma[i] == +1) if is_pos else (sigma[i] == -1)
        sat = sat or lit
        if sat: break
    return 0 if sat else 1

def unsat_count(cnf, sigma):
    return sum(clause_unsat(c, sigma) for c in cnf)

# Fázová chyba pro dané sigma (bez auto-locku, čistá geometrie toku)
def delta_phi(cnf, sigma, Phi_base, Phi_unit, x, t, omega, q=Q_E):
    u = unsat_count(cnf, sigma)
    #print("UNSAT clauses u =", u)
    Phi0 = (2*math.pi*HBAR)/q
    Phi  = Phi_base + u*Phi_unit
    dphi_geo = 2*math.pi*((Phi / Phi0) % 1.0)
    total = math.pi*x + omega*t + dphi_geo
    n_star = int(round(total/(2.0*math.pi)))
    return ((total - 2.0*math.pi*n_star + math.pi) % (2.0*math.pi)) - math.pi

# Jednoduchý GSAT: hledá sigma s |δφ| <= tol
def solve_sat_by_resonance(cnf, n, tol=0.02, steps=2000):
    """
    Inkrementální verze:
    - drží počty uspokojených literálů v každé klauzuli (cl_sat_cnt)
    - u = #UNSAT aktualizuje jen podle klauzulí, kde se proměnná vyskytuje
    - pro rozhodnutí stačí delta_phi(u) (Φ0/4, k0 stejně jako u tebe)
    """
    # --- rezonance (stejné jako u tebe) ---
    x, t, omega = 0.25, 1.0, 2*math.pi
    Phi0 = (2*math.pi*HBAR)/Q_E
    Phi_unit = Phi0/4.0  # π/2 na špatnou klauzuli
    k0 = int(round((math.pi*x + omega*t)/(2.0*math.pi)))
    frac_base = (k0 - (math.pi*x + omega*t)/(2.0*math.pi)) % 1.0
    Phi_base = frac_base * Phi0

    # --- pomocné struktury pro inkrementální u ---
    # cnf je ve tvaru [( (var, is_pos), ... ), ...]
    # vyrobíme výskyty proměnných do klauzulí a znaménka (±1)
    occ_idx = [[] for _ in range(n)]
    occ_sgn = [[] for _ in range(n)]
    for ci, cl in enumerate(cnf):
        for (v, is_pos) in cl:
            occ_idx[v].append(ci)
            occ_sgn[v].append(1 if is_pos else -1)

    # náhodná počáteční sigma
    sigma = [random.choice((-1, +1)) for _ in range(n)]

    # spočítáme sat-count každé klauzule a u = #UNSAT
    cl_sat_cnt = [0]*len(cnf)
    def lit_satisfied(v, is_pos):
        return (sigma[v] == +1) if is_pos else (sigma[v] == -1)

    u = 0
    for ci, cl in enumerate(cnf):
        s = 0
        for (v, is_pos) in cl:
            if lit_satisfied(v, is_pos):
                s += 1
        cl_sat_cnt[ci] = s
        if s == 0:
            u += 1

    # rychlá delta_phi z aktuálního u (žádné přepočítávání přes celé CNF)
    def dphi_from_u(u_val):
        Phi  = Phi_base + u_val*Phi_unit
        dphi_geo = 2*math.pi*((Phi / Phi0) % 1.0)
        total = math.pi*x + omega*t + dphi_geo
        n_star = int(round(total/(2.0*math.pi)))
        return ((total - 2.0*math.pi*n_star + math.pi) % (2.0*math.pi)) - math.pi

    best = dphi_from_u(u)

    # hlavní smyčka
    for _ in range(steps):
        if abs(best) <= tol:
            return True, _, sigma, best  # hotovo

        # najdi nejlepší flip (děláme jen inkrement u)
        j_star = None
        val_star = best

        for j in range(n):
            # spočti Δu při flipu j (přes jeho výskyty)
            delta_u = 0
            new_val = -sigma[j]
            old_val = sigma[j]
            for (ci, sgn) in zip(occ_idx[j], occ_sgn[j]):
                was_sat = (old_val == +1) if (sgn > 0) else (old_val == -1)
                will_sat = (new_val == +1) if (sgn > 0) else (new_val == -1)
                s = cl_sat_cnt[ci]
                if not was_sat and will_sat:
                    # klauzule nabude 1, pokud byla 0 → u klesne o 1
                    if s == 0: delta_u -= 1
                elif was_sat and not will_sat:
                    # klauzule ztratí 1; pokud byla 1 → spadne na 0 → u +1
                    if s == 1: delta_u += 1

            cand = dphi_from_u(u + delta_u)
            if abs(cand) < abs(val_star):
                val_star = cand
                j_star = j

        if j_star is None:
            # stagnace → WalkSAT kopanec
            j_star = random.randrange(n)

        # aplikuj flip j_star a aktualizuj u + cl_sat_cnt
        new_val = -sigma[j_star]
        old_val = sigma[j_star]
        sigma[j_star] = new_val

        for (ci, sgn) in zip(occ_idx[j_star], occ_sgn[j_star]):
            was_sat = (old_val == +1) if (sgn > 0) else (old_val == -1)
            will_sat = (new_val == +1) if (sgn > 0) else (new_val == -1)
            if not was_sat and will_sat:
                if cl_sat_cnt[ci] == 0:
                    u -= 1
                cl_sat_cnt[ci] += 1
            elif was_sat and not will_sat:
                cl_sat_cnt[ci] -= 1
                if cl_sat_cnt[ci] == 0:
                    u += 1

        best = dphi_from_u(u)

    return False, steps, sigma, best

def random_3sat(n: int, m: int):
    cnf = []
    for _ in range(m):
        lits = []
        vars_ = random.sample(range(n), 3) if n >= 3 else [0,1,2][:n]
        for v in vars_:
            lits.append((v, random.choice([True, False])))
        while len(lits) < 3:
            lits.append((random.randrange(max(1,n)), True))
        cnf.append(tuple(lits[:3]))
    return cnf

import time, glob

for soubor in sorted(glob.glob("u*.cnf")):

    #formula = random_3sat(n=200, m=860)
    #print(formula)
    #out = solve_sat_by_resonance(cnf=formula, n=200)
    #print(out)



    t0 = time.time()

    print(f"soubor:  {soubor}")

    n_vars, clauses = parse_dimacs_cnf(soubor)
    print(f"clauses: {clauses}")

    cnf = [dimacs_to_clause(clause) for clause in clauses]
    out = solve_sat_by_resonance(cnf, n_vars)
    print(f"out:     {out}")

    spectral_time = time.time() - t0
    print(f"time:    {spectral_time}")

    print(" ")