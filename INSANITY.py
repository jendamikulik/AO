import numpy as np
import math
import matplotlib.pyplot as plt

def popcount(x: int) -> int:
    return bin(x).count('1')

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
    pc = np.array([popcount(b) for b in bits])
    return np.where((pc & 1) == 0, 1, -1).astype(np.int8)

def truncated_hadamard(m: int, idx: int = 1) -> np.ndarray:
    if m <= 0:
        return np.zeros(1, dtype=np.int8)
    N = _next_pow2(m)
    k = idx % N
    if k == 0:
        k = 1
    row = _walsh_row(N, k)
    return row[:m]

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

# SAT instance: 3-var satisfiable
sat_clauses = [
    [1, 2, 3],
    [-1, -2, 3],
    [-1, 2, -3],
    [1, -2, -3]
]
sat_n = 3

# UNSAT instance: 3-var unsatisfiable with 8 clauses
unsat_clauses = [
    [1, 2, 3],
    [1, 2, -3],
    [1, -2, 3],
    [1, -2, -3],
    [-1, 2, 3],
    [-1, 2, -3],
    [-1, -2, 3],
    [-1, -2, -3]
]
unsat_n = 3

T = 2000  # Steps for annealing
frac = (math.sqrt(5) - 1) / 2
stride = stride_near(T, frac)
print(f"Anti-aliasing stride: {stride}")

def run_simulation(clauses, n, label, visualize_steps=[]):
    spins = np.random.choice([-1, 1], n)
    def energy(spins):
        unsat = 0
        for clause in clauses:
            sat = any(spins[abs(lit) - 1] == (1 if lit > 0 else -1) for lit in clause)
            if not sat:
                unsat += 1
        return unsat
    current_energy = energy(spins)
    temp = 5.0  # Initial annealing temp
    energies = []
    coherences = []
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
        temp *= 0.995  # Slower annealing
        energies.append(current_energy)
        phases = np.arccos(spins)
        mean_vec = np.mean(np.exp(1j * phases))
        kappa_proxy = abs(mean_vec)
        coherences.append(kappa_proxy)
        if t in visualize_steps:
            plot_phasors(spins, t, label)
            plt.pause(0.01)  # For animation effect
        if t % 200 == 0:
            print(f"{label} Step {t}: Energy {current_energy}, Coherence {kappa_proxy:.4f}")
    return energies, coherences, current_energy, kappa_proxy, spins

def plot_phasors(spins, step, label):
    phases = np.arccos(spins)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    circle = plt.Circle((0, 0), 1, color='b', fill=False)
    ax.add_artist(circle)
    for i, phase in enumerate(phases):
        ax.arrow(0, 0, np.cos(phase), np.sin(phase), head_width=0.05, head_length=0.1, fc='r', ec='r')
        ax.text(np.cos(phase) * 1.1, np.sin(phase) * 1.1, f'Var {i+1}')
    ax.set_title(f'{label} Fázory na jednotkové kružnici - Krok {step}')
    plt.show(block=False)

# Run simulations with visualization on selected steps (e.g., every 100)
visualize_steps = list(range(0, T, 100))  # Adjust for less/more plots

print("Running SAT simulation...")
sat_energies, sat_coherences, sat_final_e, sat_final_k, sat_final_spins = run_simulation(sat_clauses, sat_n, "SAT", visualize_steps)
print(f"SAT Final: Energy {sat_final_e}, Coherence {sat_final_k:.4f}")

print("\nRunning UNSAT simulation...")
unsat_energies, unsat_coherences, unsat_final_e, unsat_final_k, unsat_final_spins = run_simulation(unsat_clauses, unsat_n, "UNSAT", visualize_steps)
print(f"UNSAT Final: Energy {unsat_final_e}, Coherence {unsat_final_k:.4f}")

# Plot energy and coherence comparisons
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].plot(sat_energies)
axs[0, 0].set_title('SAT Energie')
axs[0, 1].plot(sat_coherences)
axs[0, 1].set_title('SAT Koherence')
axs[1, 0].plot(unsat_energies)
axs[1, 0].set_title('UNSAT Energie')
axs[1, 1].plot(unsat_coherences)
axs[1, 1].set_title('UNSAT Koherence')
plt.tight_layout()
plt.show()