# Extended demo: add breathing (normal rim motion) and conveyor phase,
# and show small-signal linear signatures predicted by AO.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def conveyor_phase(L_eff, u, lamb, c=C):
    # per full loop conveyor phase (two directions difference); for single direction,
    # we add half. For interference contrast we add the full effective difference per loop
    # to the accumulated phase count m (as with Sagnac).
    return 4*np.pi*L_eff*u/(lamb*c)

def fringes(R, theta_in, Omega, M, lambda0, phi0=0.0, rho=0.985,
            Vrad_t=lambda t: 0.0*t, A_eff=None, L_eff=None, u=0.0):
    s, phi = ring_map(R, theta_in, phi0, M)
    # arrival times for each segment end (cumulative)
    t = np.cumsum(np.full(M, s/C))
    # normal velocity sequence from breathing/roughness
    v_n = Vrad_t(t)
    lam = lambda_update(lambda0, v_n)
    # base phases from optical path
    phases = np.array([phase_sum(m, s, lam) for m in range(1, M+1)], dtype=float)
    # add rotation Sagnac per loop
    if A_eff is not None and Omega != 0.0:
        phases += np.array([m * sagnac_phase(A_eff, Omega, lambda0) for m in range(1, M+1)])
    # add conveyor phase per loop
    if L_eff is not None and u != 0.0:
        phases += np.array([m * conveyor_phase(L_eff, u, lambda0) for m in range(1, M+1)])
    # amplitude weights (uniform rho^(m-1)); can be generalized for asymmetry
    Aq = rho**np.arange(M)
    E = np.sum(Aq * np.exp(1j*phases))
    I = np.abs(E)**2
    return I

# Parameters
R = 0.1                  # m
lambda0 = 1550e-9        # m
rho = 0.985
M = 120
phi0 = 0.0
theta_fix = 0.08         # radians
A_eff = np.pi * R**2
# Loop length approximation for conveyor
L_eff = (2*np.pi*R)/np.sin(theta_fix)

# Choose Omega so baseline is near a fringe slope (enhances sensitivity)
Omega_base = 35.0  # rad/s (illustrative)

# 1) Breathing test: Vrad(t) = A * 2π f * cos(2π f t)
f_breath = 1_000.0  # Hz breathing frequency
def Vrad_of_t_factory(A):
    return lambda t: A * (2*np.pi*f_breath) * np.cos(2*np.pi*f_breath * t)

A_vals = np.linspace(0.0, 0.5, 30)  # amplitude in meters (!) of boundary displacement; A*2πf ~ v_n
# The resulting normal velocities are v_n ~ A*2πf; we will also report peak v_n for clarity
I_breath = []
v_n_peak = []
for A in A_vals:
    Vrad_t = Vrad_of_t_factory(A)
    # peak normal speed for this A
    v_peak = A * 2*np.pi*f_breath
    v_n_peak.append(v_peak)
    I = fringes(R, theta_fix, Omega_base, M, lambda0, phi0, rho,
                Vrad_t=Vrad_t, A_eff=A_eff, L_eff=None, u=0.0)
    I_breath.append(I)
I_breath = np.array(I_breath)
v_n_peak = np.array(v_n_peak)

# Fit small-signal linear relation I vs v_n_peak near zero amplitude
mask_lin = v_n_peak <= (0.5 * v_n_peak.max()/5 + 1e-12)  # take the first few small values
if np.count_nonzero(mask_lin) >= 5:
    coeffs_breath = np.polyfit(v_n_peak[mask_lin], I_breath[mask_lin], 1)
else:
    coeffs_breath = np.polyfit(v_n_peak[:5], I_breath[:5], 1)

plt.figure()
plt.plot(v_n_peak, I_breath, 'o', label='data')
# plot linear fit over the same range
xfit = np.linspace(v_n_peak.min(), v_n_peak.max(), 200)
yfit = coeffs_breath[0]*xfit + coeffs_breath[1]
plt.plot(xfit, yfit, '-', label='linear fit (small v_n)')
plt.xlabel("Peak normal speed v_n,peak (m/s)")
plt.ylabel("Intensity (arb. units)")
plt.title("Breathing test: Intensity vs peak normal speed (AO predicts linear small-signal)")
plt.legend()
plt.grid(True)
plt.savefig('breathing_linear_signature.png', dpi=150, bbox_inches='tight')
plt.show()

# 2) Conveyor test: sweep u (translation), show I vs u linear periodicity
u_vals = np.linspace(0.0, 1.5, 50)   # m/s
I_conv = []
for u in u_vals:
    I = fringes(R, theta_fix, 0.0, M, lambda0, phi0, rho,
                Vrad_t=lambda t: 0.0*t, A_eff=None, L_eff=L_eff, u=u)
    I_conv.append(I)
I_conv = np.array(I_conv)

# Rough linear fit near u=0 (small u regime)
mask_lin_u = u_vals <= 0.2
coeffs_u = np.polyfit(u_vals[mask_lin_u], I_conv[mask_lin_u], 1)

plt.figure()
plt.plot(u_vals, I_conv, 'o', label='data')
xfit_u = np.linspace(u_vals.min(), u_vals.max(), 200)
yfit_u = coeffs_u[0]*xfit_u + coeffs_u[1]
plt.plot(xfit_u, yfit_u, '-', label='linear fit (near u=0)')
plt.xlabel("Conveyor speed u (m/s)")
plt.ylabel("Intensity (arb. units)")
plt.title("Conveyor test: Intensity vs u (AO predicts linear small-signal in phase)")
plt.legend()
plt.grid(True)
plt.savefig('conveyor_linear_signature.png', dpi=150, bbox_inches='tight')
plt.show()


