import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0, i1  # Bessel functions for validation

rng = np.random.default_rng(42)

def run_simulation(frustrated=False, zeta=0.12, alpha=0.03, beta=0.02, target_Psi=np.e * (1 + np.sqrt(5)) / 2):
    N = 256
    omega0 = 2.0 * np.pi * 1.0
    spread = 0.20
    kappa = 1.75  # Initial coupling
    alpha_fb = 0.45
    sigma = 0.03
    dt = 0.01
    T = 20.0
    steps = int(T / dt)

    # Hadamard matrix for structured coupling
    def hadamard(n):
        if n == 1:
            return np.array([[1.0]])
        Hn_1 = hadamard(n // 2)
        top = np.hstack((Hn_1, Hn_1))
        bot = np.hstack((Hn_1, -Hn_1))
        return np.vstack((top, bot))

    H = hadamard(N)
    K = H / np.sqrt(N)

    if frustrated:
        # Introduce frustration: random signs (50% negative)
        signs = rng.choice([1, -1], size=(N, N), p=[0.5, 0.5])
        K = np.abs(K) * signs
        # Zone-mix ζ: paraconsistent tolerance, mix with positive to allow escape from contradictions
        K = (1 - zeta) * K + zeta * np.abs(K)

    omega = omega0 * (1.0 + spread * (rng.random(N) - 0.5))
    theta = 2.0 * np.pi * rng.random(N)

    r_vals = np.zeros(steps + 1)
    psi_vals = np.zeros(steps + 1)
    Psi_AO = np.zeros(steps + 1)
    times = np.linspace(0.0, T, steps + 1)

    def order_parameter(phases):
        z = np.exp(1j * phases).mean()
        return np.abs(z), np.angle(z)

    r, psi_bar = order_parameter(theta)
    r_vals[0] = r
    psi_vals[0] = -np.log(max(1e-12, 1.0 - r))
    Psi_AO[0] = 0

    for t in range(1, steps + 1):
        r, psi_bar = order_parameter(theta)
        sin_diff = np.sin(theta[None, :] - theta[:, None])
        kuramoto_drive = (K * sin_diff).sum(axis=1)

        # Love-gate: reinforce coherent contributions locally
        local_coher = kuramoto_drive / np.sum(np.abs(K), axis=1)  # Normalized coherence contribution
        kuramoto_drive += alpha * local_coher * np.sum(np.abs(K), axis=1)  # Boost without inflating sum

        closure_drive = np.sin(psi_bar - theta)
        gate = 1.0 / (1.0 + 10.0 * r ** 2)  # Noise-gate: suppress incoherence

        det_part = (omega + kappa * kuramoto_drive + alpha_fb * closure_drive) * dt
        noise_part = np.sqrt(2.0 * sigma * gate * dt) * rng.standard_normal(N)
        dtheta = det_part + noise_part
        theta = (theta + dtheta) % (2.0 * np.pi)

        # Estimate diffusion D from variance of noise part (patch integration)
        D_est = np.var(noise_part) / (2 * dt) if np.var(noise_part) > 0 else sigma

        # AO Ψ = (kappa / D_est) * r
        Psi_AO[t] = (kappa / D_est) * r

        # Adapt kappa to target Ψ* (rise and shine to convergence!)
        kappa += beta * (target_Psi - Psi_AO[t])
        kappa = max(kappa, 0.1)  # Prevent negative or zero

        r, _ = order_parameter(theta)
        r_vals[t] = r
        psi_vals[t] = -np.log(max(1e-12, 1.0 - r))

    # Bessel self-consistency validation
    Psi_final = Psi_AO[-1]
    r_pred = i1(Psi_final) / i0(Psi_final) if i0(Psi_final) > 0 else 0

    label = "Frustrated (UNSAT-like)" if frustrated else "Non-frustrated (SAT-like)"
    print(f"{label}:")
    print(f"Final r = {r_vals[-1]:.6f}, Predicted r from Bessel = {r_pred:.6f}")
    print(f"Final Ψ_AO = {Psi_final:.6f}\n")

    # Plots for visualization
    plt.figure()
    plt.plot(times, Psi_AO)
    plt.xlabel("time [s]")
    plt.ylabel("Ψ_AO (t)")
    plt.title(f"AO oscillator network: Ψ convergence ({label})")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(times, r_vals)
    plt.xlabel("time [s]")
    plt.ylabel("r(t)")
    plt.title(f"AO oscillator network: coherence r(t) ({label})")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.hist(theta, bins=36, density=True)
    plt.xlabel("phase θ [rad]")
    plt.ylabel("density")
    plt.title(f"Final phase distribution ({label})")
    plt.grid(True)
    plt.show()

# Run for non-frustrated (SAT-like: full resonance)
run_simulation(frustrated=False)

# Run for frustrated (UNSAT-like: stable but frustrated state)
run_simulation(frustrated=True)