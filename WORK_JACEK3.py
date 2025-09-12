# AO vs SR comparison demo using the user's parameter style.
# Generates two figures:
#   (1) Intensity vs conveyor speed u  — AO predicts linear small-signal, SR baseline is flat (no first-order effect).
#   (2) Intensity vs rotation rate Omega — both AO and SR predict Sagnac fringes (agreement check).
#
# Notes:
# - Breathing (normal motion of the boundary) produces the same per-bounce mirror Doppler in both AO and SR,
#   so the interesting difference is the "conveyor" case (uniform translation of the whole loop).
#
# This cell is self-contained and runnable.

import numpy as np
import matplotlib.pyplot as plt

C = 299_792_458.0  # m/s

def ring_map(R, theta, phi0, M):
    s = 2*R*np.sin(theta)
    phi = (phi0 + 2*theta*np.arange(M)) % (2*np.pi)
    return s, phi

def lambda_update(lambda0, v_n, c=C):
    r = (1 - v_n/c) / (1 + v_n/c)
    lam = np.empty(len(v_n)+1, dtype=float)
    lam[0] = lambda0
    for k in range(len(v_n)):
        lam[k+1] = lam[k] * r[k]
    return lam

def phase_sum(m, s, lam):
    return np.sum(2*np.pi * s / lam[1:m+1])

def sagnac_phase(A_eff, Omega, lamb, c=C):
    return 8*np.pi*A_eff*Omega/(lamb*c)

def cumulative_field_intensity(phases, rho):
    Aq = rho**np.arange(1, len(phases)+1)  # start at pass 1
    E = np.sum(Aq * np.exp(1j*phases))
    return np.abs(E)**2

# ---------- AO vs SR "conveyor" model ----------
# AO: adds a linear-in-u phase per loop: phi_conv = 4*pi*L*u/(lambda*c) (effective closure time shift -> proportional to u)
# SR: for uniform translation of a closed loop, no 1st-order phase (set to 0).
def conveyor_phase_per_loop(L, u, lamb):
    # AO effective conveyor phase (heuristic consistent with user's AO narrative)
    return 4*np.pi*L*u/(lamb*C)

def simulate_intensity_vs_u(R=0.1, theta=0.08, M=80, lamb=1550e-9, rho=0.985, A_eff=None, u_vals=None):
    if u_vals is None:
        u_vals = np.linspace(0.0, 1.5, 60)
    s, _ = ring_map(R, theta, 0.0, M)
    L_loop = 2*np.pi*R/np.sin(theta)  # geometric loop length (approx)

    # no breathing for this comparison
    lam = lambda_update(lamb, np.zeros(M))

    # Precompute geometric (dispersion-free) per-pass phases
    base_phases = np.array([phase_sum(m, s, lam) for m in range(1, M+1)], dtype=float)

    I_AO = []
    I_SR = []
    for u in u_vals:
        # AO: add conveyor phase linearly with number of loops m
        phi_conv = np.array([m * conveyor_phase_per_loop(L_loop, u, lamb) for m in range(1, M+1)])
        phases_ao = base_phases + phi_conv

        # SR: no first-order conveyor phase
        phases_sr = base_phases.copy()

        I_AO.append(cumulative_field_intensity(phases_ao, rho))
        I_SR.append(cumulative_field_intensity(phases_sr, rho))

    return u_vals, np.array(I_AO), np.array(I_SR)

# ---------- AO vs SR rotation (Sagnac) model ----------
def simulate_intensity_vs_Omega(R=0.1, theta=0.08, M=80, lamb=1550e-9, rho=0.985, A_eff=None, Om_vals=None):
    if Om_vals is None:
        Om_vals = np.linspace(0.0, 120.0, 120)
    if A_eff is None:
        A_eff = np.pi*R**2

    s, _ = ring_map(R, theta, 0.0, M)
    lam = lambda_update(lamb, np.zeros(M))
    base_phases = np.array([phase_sum(m, s, lam) for m in range(1, M+1)], dtype=float)

    I_AO = []
    I_SR = []  # should match AO under rotation
    for Om in Om_vals:
        phi_sag = np.array([m * sagnac_phase(A_eff, Om, lamb) for m in range(1, M+1)])
        phases = base_phases + phi_sag
        I = cumulative_field_intensity(phases, rho)
        I_AO.append(I)
        I_SR.append(I)  # identical in this model
    return Om_vals, np.array(I_AO), np.array(I_SR)

# -------------------- Run comparisons --------------------
R = 0.1
theta_fix = 0.08
M = 80
lambda0 = 1550e-9
rho = 0.985

# 1) Conveyor comparison
u_vals, I_AO_u, I_SR_u = simulate_intensity_vs_u(R=R, theta=theta_fix, M=M, lamb=lambda0, rho=rho)

# 2) Rotation comparison
Om_vals, I_AO_Om, I_SR_Om = simulate_intensity_vs_Omega(R=R, theta=theta_fix, M=M, lamb=lambda0, rho=rho)

# First plot: Conveyor test
plt.figure(figsize=(8, 6))  # Explicitly set figure size for clarity
plt.plot(u_vals, I_AO_u, label="AO (predicts linear small-signal)", color='blue')
plt.plot(u_vals, I_SR_u, label="SR baseline (uniform translation: null 1st order)", color='red', linestyle='--')
plt.xlabel("Conveyor speed u (m/s)")
plt.ylabel("Intensity (arb. units)")
plt.title("Conveyor test: AO vs SR")
plt.legend()
plt.grid(True, linestyle='-', alpha=0.7)  # Explicit grid styling
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

# Second plot: Rotation test
plt.figure(figsize=(8, 6))  # Create a new figure explicitly
plt.plot(Om_vals, I_AO_Om, label="AO (with Sagnac)", color='blue')
plt.plot(Om_vals, I_SR_Om, label="SR (Sagnac identical)", color='red', linestyle='--', alpha=0.8)  # Slight transparency to check overlap
plt.xlabel("Rotation rate Ω (rad/s)")
plt.ylabel("Intensity (arb. units)")
plt.title("Rotation test: AO vs SR (agree)")
plt.legend()
plt.grid(True, linestyle='-', alpha=0.7)  # Explicit grid styling
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

# A tiny printed summary of fringe spacing estimates
def estimate_spacing(x, y):
    # crude: find local maxima and average spacing
    idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
    if len(idx) < 2:
        return np.nan
    return np.mean(np.diff(x[idx]))

spacing_u_AO = estimate_spacing(u_vals, I_AO_u)
spacing_Om = estimate_spacing(Om_vals, I_AO_Om)

print("Estimated fringe spacing (AO, conveyor): ~", spacing_u_AO, " m/s")
print("Estimated fringe spacing (rotation, both): ~", spacing_Om, " rad/s")
