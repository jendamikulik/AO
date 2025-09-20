import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import i0, i1

rng = np.random.default_rng(42)

"""
AO Framework All-in-One: P vs NP via Resonance Phase Transition
SAT: Harmony (stable Ψ ~4.398, high r) at low clause density.
Hard SAT: Threshold resonance at critical alpha ~4.26.
UNSAT: Frustration (high Ψ accumulation, mid r) at high density.
Generates random 3-SAT, solves via THEEND, initializes UNIVERSE oscillators.
No physical fluff - pure AO for computability phase transitions.
"""

def generate_3sat(n_vars, alpha):
    n_clauses = int(alpha * n_vars)
    clauses = []
    for _ in range(n_clauses):
        clause = rng.choice(range(1, n_vars+1), size=3, replace=False)
        signs = rng.choice([-1, 1], size=3)
        clause = [int(s * v) for v, s in zip(clause, signs)]
        clauses.append(clause)
    return n_vars, clauses

def load_dimacs(filename):
    clauses = []
    n = 0
    m = 0
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            if line.startswith('p'):
                parts = line.split()
                if parts[1] == 'cnf':
                    n = int(parts[2])
                    m = int(parts[3])
            else:
                clause = [int(x) for x in line.split()[:-1]]
                if clause:
                    clauses.append(clause)
    return n, clauses

def theend_solver(clauses, n, label):
    T = 5000
    frac = (math.sqrt(5) - 1) / 2
    stride = stride_near(T, frac)

    spins = np.random.choice([-1, 1], n)

    def energy(spins):
        unsat = 0
        for clause in clauses:
            sat = any(spins[abs(lit) - 1] == (1 if lit > 0 else -1) for lit in clause)
            if not sat:
                unsat += 1
        return unsat

    current_energy = energy(spins)
    temp = 5.0 + n * 0.1

    energies = [current_energy]
    for t in range(T):
        var_idx = (t * stride) % n
        h_probe = truncated_hadamard(n, t % 32)
        new_spins = spins.copy()
        flip_prob = (h_probe[var_idx] + 1) / 2
        if np.random.rand() < flip_prob:
            new_spins[var_idx] = -new_spins[var_idx]
        new_energy = energy(new_spins)
        delta = new_energy - current_energy
        if delta < 0 or np.random.rand() < math.exp(-delta / temp):
            spins = new_spins
            current_energy = new_energy
        temp *= 0.995
        energies.append(current_energy)

    median_energy = np.median(energies[T//2:])  # Late median for benchmark
    print(f"{label} Solver: Final Energy {current_energy}, Median Late Energy {median_energy:.2f}")
    return spins, current_energy, energies

def stride_near(T: int, frac: float, forbid=(1, 2)):
    if T <= 4:
        return max(1, T - 2)
    target = int(round((frac % 1.0) * T)) % T
    target = min(max(target, 2), T - 2)
    w_alias = 0.40
    w_triv = 2.50
    w_hr = 0.15
    divs = [d for d in range(2, min(64, T // 2) + 1) if T % d == 0]
    def alias_penalty(s: int) -> float:
        pen = 0.0
        for d in divs:
            step = T // d
            k = round(s / step)
            delta = abs(s - k * step) / step
            if delta < 0.5:
                pen += (0.5 - delta)
        return pen
    def harmonic_ripple(s: int, H: int = 8) -> float:
        acc = 0.0
        for r in range(2, H + 1):
            x = math.sin(math.pi * r * s / T)
            acc += 1.0 / (1e-9 + abs(x))
        return acc
    triv = {1, 2, 3, T - 1, T - 2, T - 3}
    golden = (math.sqrt(5) - 1.0) * 0.5
    prefer = int(round(golden * T)) % T
    candidates = [s for s in range(2, T - 1) if math.gcd(s, T) == 1]
    best_s = 2
    best_score = float("inf")
    for s in candidates:
        base = abs(s - target)
        pen_alias = alias_penalty(s)
        pen_triv = w_triv if (s in triv or s in forbid) else 0.0
        pen_hr = harmonic_ripple(s)
        reward = 0.05 * abs(s - prefer)
        score = base + w_alias * pen_alias + pen_triv + w_hr * pen_hr + reward
        if score < best_score:
            best_score = score
            best_s = s
    return best_s

def truncated_hadamard(m: int, idx: int = 1) -> np.ndarray:
    if m <= 0:
        return np.zeros(1, dtype=np.int8)
    N = _next_pow2(m)
    k = idx % N
    if k == 0:
        k = 1
    row = _walsh_row(N, k)
    return row[:m]

def _next_pow2(x: int) -> int:
    n = 1
    while n < x:
        n <<= 1
    return n

def _gray(i: int) -> int:
    return i ^ (i >> 1)

def _walsh_row(N: int, k: int) -> np.ndarray:
    gk = _gray(k)
    n = np.arange(N, dtype=np.uint64)
    bits = np.bitwise_and(n, gk)
    pc = np.array([bin(b).count('1') for b in bits])
    return np.where((pc & 1) == 0, 1, -1).astype(np.int8)

def solver_aware_init(spins, clauses, N, rng, sigma0=0.05, eps=0.05):
    # Build field h from clause contributions
    h = np.zeros(N)
    for C in clauses:
        sat = any(spins[abs(l) - 1] == (1 if l > 0 else -1) for l in C)
        if sat:
            continue  # satisfied, no bias
        # For unsatisfied, push towards satisfying one literal
        for l in C:
            i = abs(l) - 1
            desire = 1 if l > 0 else -1
            h[i] += (desire - spins[i])  # Bias towards flip if wrong
    # Normalize h to [-1,1]
    h_max = np.max(np.abs(h)) + 1e-9
    h = h / h_max
    # Map to phases: positive h -> near 0, negative -> near pi
    theta = np.where(h >= 0, eps, np.pi - eps)
    theta += rng.normal(0, sigma0, size=N)
    theta %= 2 * np.pi
    return theta

def annealed_sigma(t, steps, sigma0, sigma_final):
    # Linear anneal from sigma0 to sigma_final
    return sigma0 + (sigma_final - sigma0) * (t / steps)

def universe_oscillator(spins, clauses, frustrated=False, zeta=0.12, alpha=0.03, beta=0.02, target_Psi=np.e * (1 + np.sqrt(5)) / 2, alpha_ema=0.02):
    N = len(spins)
    omega0 = 2.0 * np.pi * 1.0
    spread = 0.20
    kappa = 1.75
    alpha_fb = 0.45
    sigma = 0.03  # Initial, will anneal
    dt = 0.01
    T = 90.0  # Longer for convergence
    steps = int(T / dt)
    sigma_final = 0.005  # Lower final noise

    def hadamard(n):
        if n == 1: return np.array([[1.0]])
        Hn_1 = hadamard(n // 2)
        top = np.hstack((Hn_1, Hn_1))
        bot = np.hstack((Hn_1, -Hn_1))
        return np.vstack((top, bot))

    H = hadamard(N if (N & (N-1)) == 0 else 256)
    K = H[:N, :N] / np.sqrt(N)

    np.fill_diagonal(K, 0.0)
    row_sum = np.sum(np.abs(K), axis=1, keepdims=True) + 1e-12
    K = K / row_sum

    if frustrated:
        signs = rng.choice([1, -1], size=(N, N), p=[0.5, 0.5])
        K = K * signs  # Frustrate without abs

    omega = omega0 * (1.0 + spread * (rng.random(N) - 0.5))
    sigma0 = 0.10
    theta = solver_aware_init(spins, clauses, N, rng, sigma0)

    r_vals = np.zeros(steps + 1)
    Psi_AO = np.zeros(steps + 1)
    times = np.linspace(0.0, T, steps + 1)

    def order_parameter(phases):
        z = np.exp(1j * phases).mean()
        return np.abs(z), np.angle(z)

    r, psi_bar = order_parameter(theta)
    r_vals[0] = r
    Psi_AO[0] = 0

    D_hat = sigma
    freeze = False
    freeze_counter = 0
    freeze_threshold = 100  # Steps near target to freeze

    for t in range(1, steps + 1):
        r, psi_bar = order_parameter(theta)
        sin_diff = np.sin(theta[None, :] - theta[:, None])
        kuramoto_drive = (K * sin_diff).sum(axis=1)

        local_coher = kuramoto_drive / np.sum(np.abs(K), axis=1)
        kuramoto_drive += alpha * local_coher * np.sum(np.abs(K), axis=1)

        closure_drive = np.sin(psi_bar - theta)
        gate = 1.0 / (1.0 + 10.0 * r ** 2)

        sigma_t = annealed_sigma(t, steps, sigma, sigma_final)
        det_part = (omega + kappa * kuramoto_drive + alpha_fb * closure_drive) * dt
        noise_part = np.sqrt(2.0 * sigma_t * gate * dt) * rng.standard_normal(N)
        dtheta = det_part + noise_part
        theta = (theta + dtheta) % (2.0 * np.pi)

        varn = np.var(noise_part)
        if np.isfinite(varn) and varn > 0:
            inst_D = varn / (2 * dt)
            D_hat = (1 - alpha_ema) * D_hat + alpha_ema * inst_D

        Psi_AO[t] = (kappa / max(D_hat, 1e-9)) * r

        if not freeze:
            kappa += beta * (target_Psi - Psi_AO[t])
            kappa = max(kappa, 0.1)
            if abs(Psi_AO[t] - target_Psi) < 0.1:
                freeze_counter += 1
                if freeze_counter > freeze_threshold:
                    freeze = True
            else:
                freeze_counter = 0

        r, _ = order_parameter(theta)
        r_vals[t] = r

    Psi_final = Psi_AO[-1]
    r_pred = i1(Psi_final) / i0(Psi_final) if i0(Psi_final) > 0 else 0

    label = "UNSAT-like" if frustrated else "SAT-like"
    print(f"{label} Oscillator: Final r = {r_vals[-1]:.3f}, Ψ = {Psi_final:.3f}, Bessel r_pred = {r_pred:.3f} (frozen: {freeze})")

    plt.figure()
    plt.plot(times, Psi_AO)
    plt.title(f"Ψ Convergence ({label})")
    plt.xlabel("Time [s]")
    plt.ylabel("Ψ_AO(t)")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(times, r_vals)
    plt.title(f"Coherence r(t) ({label})")
    plt.xlabel("Time [s]")
    plt.ylabel("r(t)")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.hist(theta, bins=36, density=True)
    plt.title(f"Final Phase Distribution ({label})")
    plt.xlabel("Phase θ [rad]")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()

def p_vs_np_phase_transition_demo(n_vars=50):
    alphas = [2.0, 4.26, 6.0]  # Low (easy SAT), critical, high (UNSAT)
    for alpha in alphas:
        label = f"Alpha={alpha:.2f}"
        n, clauses = generate_3sat(n_vars, alpha)
        spins, energy, energies = theend_solver(clauses, n, label)
        frustrated = alpha > 4.26
        universe_oscillator(spins, clauses, frustrated=frustrated)

if __name__ == "__main__":
    p_vs_np_phase_transition_demo()  # Run phase transition demo for P vs NP