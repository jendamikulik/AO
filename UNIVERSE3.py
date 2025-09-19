import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv  # Modified Bessel functions

rng = np.random.default_rng(42)

# Parameters for frustrated system (analog to UNSAT)
N = 256  # Number of oscillators
omega0 = 2.0 * np.pi * 1.0
spread = 0.20
kappa_base = 2.0  # Base coupling strength
alpha_fb = 0.45
sigma = 0.03
dt = 0.01
T = 30.0
steps = int(T / dt)

# Introduce frustration: Mix positive and negative couplings
# For "frustrated" mode, some K_ij are negative (like spin glass)
frustration_level = 0.5  # Fraction of negative couplings
is_pow2 = (N & (N - 1)) == 0
if is_pow2:
    def hadamard(n):
        if n == 1:
            return np.array([[1.0]])
        Hn_1 = hadamard(n // 2)
        top = np.hstack((Hn_1, Hn_1))
        bot = np.hstack((Hn_1, -Hn_1))
        return np.vstack((top, bot))


    H = hadamard(N)
    K = H / np.sqrt(N)
else:
    K = np.ones((N, N)) / N

# Apply frustration: Randomly flip signs for fraction of links
flip_mask = rng.random((N, N)) < frustration_level
K[flip_mask] = -K[flip_mask]  # Negative couplings introduce frustration

omega = omega0 * (1.0 + spread * (rng.random(N) - 0.5))
theta = 2.0 * np.pi * rng.random(N)

r_vals = np.zeros(steps + 1)
psi_vals = np.zeros(steps + 1)
times = np.linspace(0.0, T, steps + 1)


def order_parameter(phases):
    z = np.exp(1j * phases).mean()
    return np.abs(z), np.angle(z)


r, psi_bar = order_parameter(theta)
r_vals[0] = r
psi_vals[0] = -np.log(max(1e-12, 1.0 - r))

# Add zone-mix zeta and love-gate
zeta = 0.12  # Paraconsistent zone-mix (small tolerance for contradictions)
love_beta = 0.02  # Love-gate strength (boost coherent parts)

for t in range(1, steps + 1):
    r, psi_bar = order_parameter(theta)
    sin_diff = np.sin(theta[None, :] - theta[:, None])
    kuramoto_drive = (K * sin_diff).sum(axis=1)

    # Closure feedback with zone-mix: Add zeta * kappa * r as "battery" to tolerate frustration
    closure_drive = np.sin(psi_bar - theta) + zeta * kappa_base * r  # Paraconsistent boost

    # Love-gate: Boost drive for coherent oscillators (where |theta_i - psi_bar| < pi/2)
    coherent_mask = np.abs(theta - psi_bar) < np.pi / 2
    kuramoto_drive[coherent_mask] *= (1 + love_beta)

    # Noise-gating
    gate = 1.0 / (1.0 + 10.0 * r ** 2)

    dtheta = (omega + kappa_base * kuramoto_drive + alpha_fb * closure_drive) * dt + np.sqrt(
        2.0 * sigma * gate * dt) * rng.standard_normal(N)
    theta = (theta + dtheta) % (2.0 * np.pi)

    r, _ = order_parameter(theta)
    r_vals[t] = r
    psi_vals[t] = -np.log(max(1e-12, 1.0 - r))

# Estimate D from phase increments (for true Psi)
dtheta_hist = []  # Collect dtheta for Var estimation
# ... (simplified, assume D = sigma for demo)

# Bessel self-consistency check for Psi_star ~ 4.398
psi_star = np.exp(1) * (1 + np.sqrt(5)) / 2  # e * phi
r_from_bessel = iv(1, psi_star) / iv(0, psi_star)
print(f"Bessel r for Psi_star={psi_star:.3f}: {r_from_bessel:.3f}")

# Print summary
Psi_final = psi_vals[-1]
r_final = r_vals[-1]
print(f"[Frustrated] Final r = {r_final:.6f}, Final Ψ = {Psi_final:.6f}")

# For comparison, run non-frustrated version
# Reset K to positive
K = np.abs(K)  # Remove frustration

theta = 2.0 * np.pi * rng.random(N)
r, psi_bar = order_parameter(theta)
r_vals[0] = r
psi_vals[0] = -np.log(max(1e-12, 1.0 - r))

for t in range(1, steps + 1):
    r, psi_bar = order_parameter(theta)
    sin_diff = np.sin(theta[None, :] - theta[:, None])
    kuramoto_drive = (K * sin_diff).sum(axis=1)
    closure_drive = np.sin(psi_bar - theta) + zeta * kappa_base * r
    coherent_mask = np.abs(theta - psi_bar) < np.pi / 2
    kuramoto_drive[coherent_mask] *= (1 + love_beta)
    gate = 1.0 / (1.0 + 10.0 * r ** 2)
    dtheta = (omega + kappa_base * kuramoto_drive + alpha_fb * closure_drive) * dt + np.sqrt(
        2.0 * sigma * gate * dt) * rng.standard_normal(N)
    theta = (theta + dtheta) % (2.0 * np.pi)
    r, _ = order_parameter(theta)
    r_vals[t] = r
    psi_vals[t] = -np.log(max(1e-12, 1.0 - r))

Psi_final_nonfrust = psi_vals[-1]
r_final_nonfrust = r_vals[-1]
print(f"[Non-frustrated] Final r = {r_final_nonfrust:.6f}, Final Ψ = {Psi_final_nonfrust:.6f}")