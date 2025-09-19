# AO Oscillator Network Simulator: Convergence of Ψ
# - Coupled phase oscillators (Kuramoto-style) with an AO "closure" feedback
# - Order parameter r(t) := |(1/N) sum_i exp(i θ_i(t))|
# - We report Ψ(t) := -log(1 - r(t))  (monotone map; Ψ → ∞ as r → 1)
# - Goal: show robust convergence of Ψ(t) under AO-style feedback and noise-gating
#
# NOTE: This is a self-contained, runnable cell. It produces three figures:
#   (1) Ψ(t) vs time
#   (2) r(t) vs time
#   (3) Final phase histogram
#
# You can tweak N, coupling kappa, noise sigma, and the feedback strength.
#
# IMPORTANT plotting rule: single-plot figures, no explicit colors.


import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# -------------------- Model parameters --------------------
N        = 256          # number of oscillators
omega0   = 2.0*np.pi*1.0   # base frequency [rad/s]
spread   = 0.20             # relative spread of natural frequencies
kappa    = 1.75             # coupling strength
alpha_fb = 0.45             # AO "closure" feedback strength
sigma    = 0.03             # noise level (phase diffusion)
dt       = 0.01             # time step [s]
T        = 20.0             # total time [s]
steps    = int(T/dt)

# Optional: Hadamard-like coupling structure (binary ±1 pattern)
def hadamard(n):
    """Recursive Walsh-Hadamard (n must be power of 2)."""
    if n == 1:
        return np.array([[1.0]])
    Hn_1 = hadamard(n//2)
    top  = np.hstack((Hn_1, Hn_1))
    bot  = np.hstack((Hn_1, -Hn_1))
    return np.vstack((top, bot))

# Ensure N is power of 2 for simple H; otherwise fall back to dense mean-field
is_pow2 = (N & (N-1)) == 0
if is_pow2:
    H = hadamard(N)
    K = H / np.sqrt(N)  # orthogonal ±1 coupling; normalized
else:
    K = np.ones((N, N)) / N  # mean-field fallback

# Natural frequencies with small heterogeneity
omega = omega0 * (1.0 + spread * (rng.random(N) - 0.5))

# Initial phases
theta = 2.0*np.pi * rng.random(N)

# Storage
r_vals  = np.zeros(steps+1)
psi_vals = np.zeros(steps+1)
times   = np.linspace(0.0, T, steps+1)

def order_parameter(phases):
    """Return Kuramoto order parameter magnitude r and mean angle psi_bar."""
    z = np.exp(1j * phases).mean()
    return np.abs(z), np.angle(z)

# Initialize metrics
r, psi_bar = order_parameter(theta)
r_vals[0]  = r
psi_vals[0] = -np.log(max(1e-12, 1.0 - r))

# AO-style feedback: (i) standard Kuramoto coupling; (ii) closure nudge toward mean phase
#                    (iii) noise-gating: reduce effective noise as coherence improves
for t in range(1, steps+1):
    r, psi_bar = order_parameter(theta)

    # Standard Kuramoto term using structured coupling matrix K
    # Effective driving on oscillator i is sum_j K_ij sin(θ_j - θ_i)
    sin_diff = np.sin(theta[None, :] - theta[:, None])
    kuramoto_drive = (K * sin_diff).sum(axis=1)

    # Closure feedback: nudge phases toward the current mean phase psi_bar (AO "phase-lock" tendency)
    closure_drive = np.sin(psi_bar - theta)

    # Noise-gating: shrink noise as r grows (strong coherence => less accepted noise)
    gate = 1.0 / (1.0 + 10.0 * r**2)

    dtheta = (omega
              + kappa * kuramoto_drive
              + alpha_fb * closure_drive) * dt \
             + np.sqrt(2.0 * sigma * gate * dt) * rng.standard_normal(N)

    theta = (theta + dtheta) % (2.0*np.pi)

    # Store metrics
    r, _ = order_parameter(theta)
    r_vals[t]   = r
    psi_vals[t] = -np.log(max(1e-12, 1.0 - r))

# -------------------- Plots --------------------
plt.figure()
plt.plot(times, psi_vals)
plt.xlabel("time [s]")
plt.ylabel("Psi(t) = -log(1 - r(t))")
plt.title("AO oscillator network: Ψ convergence")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(times, r_vals)
plt.xlabel("time [s]")
plt.ylabel("r(t) (order parameter)")
plt.title("AO oscillator network: coherence r(t)")
plt.grid(True)
plt.show()

plt.figure()
plt.hist(theta, bins=36, density=True)
plt.xlabel("phase θ [rad]")
plt.ylabel("density")
plt.title("Final phase distribution")
plt.grid(True)
plt.show()

# Print summary
Psi_final = psi_vals[-1]
r_final   = r_vals[-1]
print(f"[summary] final r = {r_final:.6f}, final Ψ = {Psi_final:.6f}")

