# ao_vs_sr_ring_demo.py
# AO vs SR comparison for a ring-bounce photon interferometer.
# Produces two figures:
#   (1) Intensity vs conveyor speed u (AO ≠ SR at O(u/c))
#   (2) Intensity vs rotation Ω (AO = SR; Sagnac)

import numpy as np
import matplotlib.pyplot as plt

# ---------------------- Physical constants ----------------------
C = 299_792_458.0  # m/s

# ---------------------- Geometry & kinematics -------------------
def ring_map(R, theta, phi0, M):
    """
    Perfect circle billiard with specular reflection.
    theta: angle from tangent at impact (radians).
    Returns chord length s and impact azimuths phi_k (not used further here,
    but included for completeness/jitter models).
    """
    s = 2 * R * np.sin(theta)                 # chord per segment
    phi = (phi0 + 2 * theta * np.arange(M)) % (2 * np.pi)  # invariant map
    return s, phi

# ---------------------- Mirror Doppler (normal only) ------------
def lambda_update(lambda0, v_n, c=C):
    """
    Per-bounce wavelength update for normal mirror velocity v_n (AO & SR agree here).
    Exact classical ratio; small-v limit gives Δf/f ≈ 2 v_n/c.
    """
    r = (1 - v_n / c) / (1 + v_n / c)         # λ_{k+1} / λ_k
    lam = np.empty(len(v_n) + 1, dtype=float)
    lam[0] = lambda0
    for k in range(len(v_n)):
        lam[k + 1] = lam[k] * r[k]
    return lam

def phase_sum(m, s, lam):
    """Cumulative optical phase for first m segments (use lam[1..m])."""
    return np.sum(2 * np.pi * s / lam[1:m + 1])

# ---------------------- Global loop phases ----------------------
def sagnac_phase(A_eff, Omega, lamb, c=C):
    """Per-loop Sagnac phase (AO = SR)."""
    return 8 * np.pi * A_eff * Omega / (lamb * c)

def conveyor_phase_AO(L_loop, u, lamb, c=C):
    """
    AO: uniform translation of the *entire closed loop* produces a linear-in-u
    closure-time phase. We model it phenomenologically as
        φ_conv ≈ κ * (L_loop / lamb) * (u / c),
    with κ = 4π giving the correct dimensions and linear scaling.
    (You can tune κ based on an experimental calibration.)
    """
    kappa = 4 * np.pi
    return kappa * (L_loop / lamb) * (u / c)

def conveyor_phase_SR(L_loop, u, lamb, c=C):
    """SR baseline: uniform translation ⇒ null to first order (φ ≈ 0 + O((u/c)^2))."""
    return 0.0

# ---------------------- Multi-pass interference -----------------
def fringes(
    R, theta_in, Omega, u, M, lambda0, rho=0.985, phi0=0.0,
    A_eff=None, model="AO", Vrad_t=lambda t: 0.0 * t
):
    """
    Returns intensity at the output after summing M passes with amplitude decay rho.
    model: "AO" or "SR"
       - Both include per-bounce normal Doppler via lambda_update (common physics).
       - Rotation: both add Sagnac.
       - Conveyor (uniform translation):
            AO adds linear-in-u loop phase; SR sets it to zero at O(u/c).
    """
    # chord and times
    s, _ = ring_map(R, theta_in, phi0, M)
    t = np.cumsum(np.full(M, s / C))  # simple, constant per-segment time
    # normal velocity sequence (breathing/roughness) — keep zero here for a clean comparison
    v_n = Vrad_t(t)
    lam = lambda_update(lambda0, v_n)

    # base per-pass phases from geometry + Doppler-updated λ
    per_pass_phase = np.array([phase_sum(m, s, lam) for m in range(1, M + 1)], dtype=float)

    # global loop phases (add linearly with the number of completed loops)
    # crude loop length for our chord model:
    L_loop = (2 * np.pi * R) / np.sin(theta_in)  # grazing approximation
    loops = np.arange(1, M + 1, dtype=float)

    sag = 0.0
    if A_eff is not None:
        sag = sagnac_phase(A_eff, Omega, lambda0) * loops  # per-loop Sagnac * (# loops)

    if model.upper() == "AO":
        conv = conveyor_phase_AO(L_loop, u, lambda0) * loops
    else:
        conv = conveyor_phase_SR(L_loop, u, lambda0) * loops

    phases = per_pass_phase + sag + conv

    # field sum with per-pass amplitude decay
    A = rho ** (loops - 1)
    E = np.sum(A * np.exp(1j * phases))
    return float(np.abs(E) ** 2)

# ---------------------- Peak picker (for spacing sanity) --------
def peaks_1d(x, y):
    idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
    return x[idx], y[idx]

# ====================== DEMO (params) ===========================
if __name__ == "__main__":
    # Geometry & optics
    R = 0.10            # 10 cm ring
    lambda0 = 1550e-9   # 1550 nm (telecom)
    rho = 0.985         # per-pass amplitude retention
    M = 80              # number of passes to sum
    phi0 = 0.0
    theta_fix = 0.08    # radians from tangent (grazing-like)
    A_eff = np.pi * R * R   # effective enclosed area for Sagnac

    # ------------------ (1) Conveyor speed u: AO vs SR ----------
    u_vals = np.linspace(0.0, 1.5, 140)  # m/s
    I_AO_u = np.array([fringes(R, theta_fix, 0.0, u, M, lambda0, rho, phi0, A_eff, model="AO") for u in u_vals])
    I_SR_u = np.array([fringes(R, theta_fix, 0.0, u, M, lambda0, rho, phi0, A_eff, model="SR") for u in u_vals])

    plt.figure()
    plt.plot(u_vals, I_AO_u, label="AO (linear-in-u conveyor phase)")
    plt.plot(u_vals, I_SR_u, label="SR baseline (null at O(u/c))")
    plt.xlabel("Conveyor speed u (m/s)")
    plt.ylabel("Intensity (arb. units)")
    plt.title("Conveyor test: AO vs SR (θ fixed)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ------------------ (2) Rotation Ω: AO vs SR ----------------
    Omega_vals = np.linspace(0.0, 120.0, 300)  # rad/s
    I_AO_Om = np.array([fringes(R, theta_fix, Om, 0.0, M, lambda0, rho, phi0, A_eff, model="AO") for Om in Omega_vals])
    I_SR_Om = np.array([fringes(R, theta_fix, Om, 0.0, M, lambda0, rho, phi0, A_eff, model="SR") for Om in Omega_vals])

    plt.figure()
    plt.plot(Omega_vals, I_AO_Om, label="AO (Sagnac)")
    plt.plot(Omega_vals, I_SR_Om, "--", label="SR (Sagnac) — identical")
    plt.xlabel("Omega (rad/s)")
    plt.ylabel("Intensity (arb. units)")
    plt.title("Rotation test: AO vs SR (θ fixed)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # --------------- Optional: print rough spacings -------------
    th_vals = np.linspace(0.01, 0.35, 300)
    I_theta = np.array([fringes(R, th, 50.0, 0.0, M, lambda0, rho, phi0, A_eff, model="AO") for th in th_vals])
    th_px, _ = peaks_1d(th_vals, I_theta)
    if len(th_px) > 1:
        dth = np.diff(th_px)
        print(f"[sanity] mean Δθ between AO peaks (θ-scan @ Ω=50 rad/s): {np.mean(dth):.4e} rad")

    # ------------------ Show figures ----------------------------
    plt.show()
