import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

N = 512  # Increased for deeper resonance
omega0 = 2.0 * np.pi * 1.0
spread = 0.15  # Slightly reduced spread for faster convergence
kappa = 2.0  # Stronger coupling
alpha_fb = 0.5  # Stronger closure feedback
sigma = 0.02  # Lower noise
dt = 0.005  # Finer time step
T = 30.0  # Longer simulation time
steps = int(T / dt)

# Hadamard coupling if pow2
is_pow2 = (N & (N-1)) == 0
if is_pow2:
    def hadamard(n):
        if n == 1:
            return np.array([[1.0]])
        Hn_1 = hadamard(n//2)
        top = np.hstack((Hn_1, Hn_1))
        bot = np.hstack((Hn_1, -Hn_1))
        return np.vstack((top, bot))
    H = hadamard(N)
    K = H / np.sqrt(N)
else:
    K = np.ones((N, N)) / N

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

for t in range(1, steps + 1):
    r, psi_bar = order_parameter(theta)
    sin_diff = np.sin(theta[None, :] - theta[:, None])
    kuramoto_drive = (K * sin_diff).sum(axis=1)
    closure_drive = np.sin(psi_bar - theta)
    gate = 1.0 / (1.0 + 10.0 * r**2)
    dtheta = (omega + kappa * kuramoto_drive + alpha_fb * closure_drive) * dt + np.sqrt(2.0 * sigma * gate * dt) * rng.standard_normal(N)
    theta = (theta + dtheta) % (2.0 * np.pi)
    r, _ = order_parameter(theta)
    r_vals[t] = r
    psi_vals[t] = -np.log(max(1e-12, 1.0 - r))

# Plots (but since no show, print summary)
Psi_final = psi_vals[-1]
r_final = r_vals[-1]
print(f"Final r = {r_final:.6f}, Final Ψ = {Psi_final:.6f}")
print("Psi convergence: initial {psi_vals[0]:.3f}, max {psi_vals.max():.3f}, final {Psi_final:.3f}")
print("Coherence r: initial {r_vals[0]:.3f}, max {r_vals.max():.3f}, final {r_final:.3f}")