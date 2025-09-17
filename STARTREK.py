# sat_holo_toy.py
# A 60-line holographic SAT demo: resonance operator + coherence projector

import math, numpy as np

# ----- CNF: (x1 v x2) & (~x1 v x2) & (x1 v ~x2)  --> SAT (x2=True)
CNF = [
    ( +1, +2),
    ( -1, +2),
    ( +1, -2),
]
N_VARS = 2

# ----- helpers
def dmod_pi(a):
    """wrap angle in [-pi, pi)"""
    x = (a + math.pi) % (2*math.pi) - math.pi
    return x

def d0(a):      # distance to 0 mod π
    y = abs(a) % math.pi
    return min(y, math.pi - y)

def dH(a):      # distance to π/2 mod π
    y = abs(a) % math.pi
    return abs(y - math.pi/2.0)

def unsat_count(cnf, sigma):
    """sigma in {+1,-1}^n, +1 means True. Count unsatisfied clauses."""
    u = 0
    for cl in cnf:
        sat = False
        for lit in cl:
            j = abs(lit) - 1
            val = (sigma[j] == +1) if (lit > 0) else (sigma[j] == -1)
            if val:
                sat = True
                break
        if not sat:
            u += 1
    return u

# ----- Resonance operator  R : u unsatisfied -> phase advance
HBAR = 1.054_571_817e-34
Q_E  = 1.602_176_634e-19
PHI0 = (2*math.pi*HBAR)/Q_E

def resonance_probe(cnf, n, eps=1e-3):
    """sample small ‘drives’ and compute median distances to {0, π/2}"""
    t, omega = 1.0, 2*math.pi
    base_unit = PHI0/4.0

    xs_main = (0.25, 0.75)
    xs = [xm+dx for xm in xs_main for dx in (-0.01, 0.0, +0.01)]
    sigmas = [np.ones(n, int), -np.ones(n, int)]
    scales = [1.0, 0.98, 1.02]

    d0s, dHs = [], []
    best = None

    for x in xs:
        # choose offset so the (x, t, omega) baseline is exactly on 2πk
        k0 = int(round((math.pi*x + omega*t)/(2*math.pi)))
        frac = (k0 - (math.pi*x + omega*t)/(2*math.pi)) % 1.0
        Phi_base = frac*PHI0

        for s in sigmas:
            sgn = int(np.sign(s[0]) or 1)
            u = unsat_count(cnf, s)
            for scl in scales:
                Phi = Phi_base + u*(base_unit*scl)
                dphi = dmod_pi(math.pi*x + omega*t + 2*math.pi*((Phi/PHI0)%1.0))
                a0, aH = d0(dphi), dH(dphi)
                d0s.append(a0); dHs.append(aH)
                cand = (min(a0,aH), dphi, x, sgn, scl)
                if best is None or cand[0] < best[0]:
                    best = cand

    med0, medH = float(np.median(d0s)), float(np.median(dHs))
    margin = medH - med0      # <0: closer to 0 -> SAT ; >0: closer to π/2 -> UNSAT
    rule = "median-zero" if margin > 0 else "median-half" if margin < 0 else "tie"
    guess = "SAT" if margin < 0 else "UNSAT" if margin > 0 else "UNKNOWN"
    info = dict(rule=rule, med_d0=med0, med_dhalf=medH, margin=margin,
                strict0=int(np.sum(np.array(d0s)<1e-2)),
                strictHalf=int(np.sum(np.array(dHs)<1e-2)))
    return guess, (best[1] if best else 0.0), info

# ----- Coherence projector  C : mask-averaged phasor
def truncated_hadamard(m, idx):
    """one row of a Walsh-Hadamard matrix, truncated to length m"""
    M = 1 << (max(1, m)-1).bit_length()
    row = np.empty(M, dtype=np.int8)
    mask = (M-1) & int(idx)
    for x in range(M):
        y, p = mask & x, 0
        while y:
            p ^= 1; y &= y-1
        row[x] = -1 if p else +1
    return row[:m]

def coherence_mu(cnf, n, L=2, rho=0.5, cR=6.0, seed=42):
    """build one schedule θ,mask and return global coherence μ"""
    rng = np.random.default_rng(seed)
    C = n
    R = max(1, int(math.ceil(cR*math.log(max(2, C)))))
    T = L*R
    m = max(1, int(math.floor(rho*T)))

    # majority assignment (cheap, deterministic)
    vote = np.zeros((T, C), np.int32)
    stride = max(1, (T//2) | 1)  # coprime-ish
    for i, cl in enumerate(cnf):
        base = truncated_hadamard(m, idx=(i*2+1)).astype(int)
        off = (i*stride) % T
        for t in range(m):
            tt = (off+t) % T
            for lit in cl:
                j = abs(lit)-1
                use_pi = (base[t] < 0) ^ (lit > 0)  # AO: sign vs literal polarity
                vote[tt, j] += (-1 if use_pi else +1)
    assign = (vote.sum(axis=0) >= 0).astype(int)  # 1=True, 0=False

    # phase field
    theta = np.zeros((T, C))
    mask  = np.zeros((T, C))
    for i, cl in enumerate(cnf):
        base = truncated_hadamard(m, idx=(i*2+1)).astype(int)
        off = (i*stride) % T
        for t in range(m):
            tt = (off+t) % T
            for lit in cl:
                j = abs(lit)-1
                sat = (assign[j]==1 and lit>0) or (assign[j]==0 and lit<0)
                use_pi = (base[t] < 0) ^ (not sat)
                theta[tt, j] = 0.0 if not use_pi else math.pi
                mask[tt,  j] = 1.0

    Z = (mask * np.exp(1j*theta)).astype(np.complex128)
    mbar = float(mask.sum(axis=0).mean())
    mu = float(abs(Z.sum())/(C*mbar)) if mbar>0 else 0.0
    return mu

# ----- Decision: resonance gate + coherence threshold
def decide(cnf, n, tau_abs=0.26, rel=3.0):
    # resonance
    r_guess, dphi, r_info = resonance_probe(cnf, n)

    # coherence (instance) and UNSAT baseline (deterministic)
    mu = coherence_mu(cnf, n, L=3, rho=0.5, cR=6.0, seed=7)
    mu_uns = coherence_mu([(+1,), (-1,)], 2, L=3, rho=0.5, cR=6.0, seed=99)  # tiny UNSAT-ish baseline
    tau = max(tau_abs, rel*mu_uns)

    # arbiter
    if r_info["margin"] <= -0.10:
        prelim, src = "SAT", "resonance"
    elif r_info["margin"] >= +0.10:
        prelim, src = "UNSAT", "resonance"
    else:
        prelim, src = ("SAT" if mu>=tau else "UNSAT"), "coherence"

    # veto: if conflict and coherence is clear
    final = prelim
    if prelim=="SAT" and mu<tau:
        final, src = "UNSAT", "coherence-veto"
    elif prelim=="UNSAT" and mu>=tau and (abs(r_info["margin"])<0.2 or r_info["strictHalf"]<=r_info["strict0"]):
        final, src = "SAT", "coherence-veto"

    return dict(guess=final, source=src, dphi=dphi, mu=mu, tau=tau, res=r_info)

if __name__ == "__main__":
    out = decide(CNF, N_VARS)
    print("Result:", out["guess"], "via", out["source"])
    print("resonance:", out["res"])
    print("coherence μ =", round(out["mu"], 4), "τ =", round(out["tau"], 4))
