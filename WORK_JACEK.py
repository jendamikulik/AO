# Demo: ring-bounce photon simulator — intensity vs. theta_in and vs. Omega
# (single-file runnable cell; generates two separate charts + a small results table)

import numpy as np
import matplotlib.pyplot as plt

# Physical constants
C = 299_792_458.0  # m/s

# ---------------------- Core functions ----------------------
def ring_map(R, theta, phi0, M):
    """Chord length s and impact angles phi_k for a circle billiard with angle from tangent theta."""
    s = 2*R*np.sin(theta)
    phi = (phi0 + 2*theta*np.arange(M)) % (2*np.pi)
    return s, phi

def vn_sequence(phi, t, Omega_t, Vrad_t, R):
    """Normal velocity at impact. Pure rotation => tangential only => no normal Doppler."""
    # Here we include only breathing/roughness via Vrad_t(t)
    return Vrad_t(t)

def lambda_update(lambda0, v_n, c=C):
    """AO per-bounce wavelength update."""
    r = (1 - v_n/c) / (1 + v_n/c)
    lam = np.empty(len(v_n)+1, dtype=float)
    lam[0] = lambda0
    for k in range(len(v_n)):
        lam[k+1] = lam[k] * r[k]
    return lam

def phase_sum(m, s, lam):
    """Cumulative optical phase for first m segments (use lam[1..m])."""
    return np.sum(2*np.pi * s / lam[1:m+1])

def sagnac_phase(A_eff, Omega, lamb, c=C):
    """Single-loop Sagnac phase shift."""
    return 8*np.pi*A_eff*Omega/(lamb*c)

def fringes(R, theta_in, Omega, M, lambda0, phi0=0.0, rho=0.98,
            Vrad_t=lambda t: 0.0*t, A_eff=None):
    """Multi-pass interference intensity at a coupler."""
    s, phi = ring_map(R, theta_in, phi0, M)
    t = np.cumsum(np.full(M, s/C))
    v_n = vn_sequence(phi, t, lambda tt: Omega+0*tt, Vrad_t, R)
    lam = lambda_update(lambda0, v_n)
    phases = np.array([phase_sum(m, s, lam) for m in range(1, M+1)], dtype=float)
    if A_eff is not None:
        # add Sagnac phase per full loop, multiplied by number of loops m
        phases += np.array([m * sagnac_phase(A_eff, Omega, lambda0) for m in range(1, M+1)])
    Aq = rho**np.arange(M)
    E = np.sum(Aq * np.exp(1j*phases))
    I = np.abs(E)**2
    return I

# ---------------------- Demo parameters ----------------------
R = 0.1            # 10 cm ring
lambda0 = 1550e-9  # 1550 nm (telecom)
rho = 0.985        # per-loop amplitude retention
M = 80             # number of passes included
phi0 = 0.0

# Effective enclosed area; for a circle we use A_eff ~ pi R^2 as a simple approximation.
A_eff = np.pi * R**2

# ---------------------- 1) Intensity vs theta_in at fixed Omega ----------------------
O_fix = 50.0  # rad/s rotation rate (illustrative)
theta_vals = np.linspace(0.01, 0.35, 300)  # radians (from tangent)
I_theta = np.array([fringes(R, th, O_fix, M, lambda0, phi0, rho, A_eff=A_eff) for th in theta_vals])

plt.figure()
plt.plot(theta_vals, I_theta)
plt.xlabel("theta_in (radians from tangent)")
plt.ylabel("Intensity (arb. units)")
plt.title("Ring-bounce interference: Intensity vs theta_in (Omega fixed)")
plt.grid(True)
plt.show()

# ---------------------- 2) Intensity vs Omega at fixed theta_in ----------------------
theta_fix = 0.08  # radians
Omega_vals = np.linspace(0.0, 120.0, 300)  # rad/s
I_Omega = np.array([fringes(R, theta_fix, Om, M, lambda0, phi0, rho, A_eff=A_eff) for Om in Omega_vals])

plt.figure()
plt.plot(Omega_vals, I_Omega)
plt.xlabel("Omega (rad/s)")
plt.ylabel("Intensity (arb. units)")
plt.title("Ring-bounce interference: Intensity vs Omega (theta fixed)")
plt.grid(True)
plt.show()

# ---------------------- 3) Small results table: peak locations ----------------------
# Extract coarse fringe spacing estimates
def peaks_1d(x, y):
    # simple peak finder: local maxima excluding edges
    idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
    return x[idx], y[idx]

th_peaks_x, th_peaks_y = peaks_1d(theta_vals, I_theta)
Om_peaks_x, Om_peaks_y = peaks_1d(Omega_vals, I_Omega)

# build a summary table as a small array; show first few peaks
import pandas as pd
n_show = 6
df = pd.DataFrame({
    "theta_peak (rad)": th_peaks_x[:n_show],
    "I(theta)_peak": th_peaks_y[:n_show],
    "Omega_peak (rad/s)": np.pad(Om_peaks_x[:n_show], (0, max(0, n_show-len(Om_peaks_x))), constant_values=np.nan),
    "I(Omega)_peak": np.pad(Om_peaks_y[:n_show], (0, max(0, n_show-len(Om_peaks_y))), constant_values=np.nan)
})

