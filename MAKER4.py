import numpy as np
import matplotlib.pyplot as plt

# ===== AO CONSTANTS =====
PI  = np.pi
PHI = (1 + np.sqrt(5)) / 2.0
LN10 = np.log(10.0)
KAPPA = (PI**2) / (PHI * LN10)          # ≈ 4.390…

# ===== CONTROLS =====
T_MAX   = 10.0
N_T     = 2000
TIME    = np.linspace(0.0, T_MAX, N_T)
DT      = TIME[1] - TIME[0]

N = 1000            # number of minds (set 1_000, 10_000, 100_000 …)
x0 = 0.015          # base microscopic leakage per mind (AO units)
sigma = 0.30        # noise level of individual minds (unitless multiplier)

# Sync program S(t): two rectangular pulses (strength times x0), adjustable
SYNC_WINDOWS = [
    (3.8, 4.2, 1.5),   # (t_start, t_end, strength_factor * x0)
    (7.8, 8.2, 0.8),
]
# Duty/strength summary for analytic comparison
duty_total = sum(max(0.0, min(T_MAX, t2) - max(0.0, t1)) for (t1,t2,_) in SYNC_WINDOWS) / T_MAX
S_bar = (sum((t2 - t1) * s for (t1,t2,s) in SYNC_WINDOWS) / max(1e-12, sum((t2 - t1) for (t1,t2,_) in SYNC_WINDOWS))) * x0 if SYNC_WINDOWS else 0.0

# ===== BUILD COLLECTIVE MODULATION =====

# Individual noise streams: mean 1, var sigma^2  (so leakage = x0 * stream)
rng = np.random.default_rng(42)
streams = 1.0 + sigma * rng.standard_normal((N, N_T))

# Law of large numbers: average across N minds
C_base = streams.mean(axis=0)

# Deterministic sync suppression S(t) in absolute leakage units
S = np.zeros_like(TIME)
for (t1, t2, s_factor) in SYNC_WINDOWS:
    mask = (TIME >= t1) & (TIME <= t2)
    S[mask] = s_factor * x0

# Effective modulation of leakage (clipped to at least 10% of x0 to avoid numeric underflow)
C_mod = np.clip(C_base, 0.1, 2.0)

# Active leakage per time
x_t = x0 * C_mod - S
x_t = np.clip(x_t, 0.0, None)   # AO cannot go negative

# ===== AO CLOSURE ACCOUNTING =====
M_t   = np.cumsum(x_t) * DT                # cumulative entropic tax (mortality)
Psi_t = np.exp(-KAPPA * M_t)               # hypersphere presence (prize)

# φ-locked bookkeeping (for the overlay)
M_harm = np.cumsum((1/PHI)    * x_t) * DT
M_esc  = np.cumsum((1/PHI**2) * x_t) * DT

# Normalizations for plot
Mort_norm  = M_t   / max(1e-12, M_t[-1])
Harm_norm  = M_harm / max(1e-12, M_t[-1])
Esc_norm   = M_esc  / max(1e-12, M_t[-1])

# ===== ANALYTIC PREDICTION =====
# Average sync strength over its support:
if SYNC_WINDOWS:
    total_sync_time = sum((t2 - t1) for (t1,t2,_) in SYNC_WINDOWS)
    S_avg_over_windows = sum((t2 - t1) * (s * x0) for (t1,t2,s) in SYNC_WINDOWS) / total_sync_time
    d = duty_total
    Sbar = S_avg_over_windows
else:
    d, Sbar = 0.0, 0.0

x_eff = max(0.0, x0 - d * (Sbar / x0) * x0)   # = x0 - d * Sbar
T_half = np.inf if x_eff == 0 else np.log(2.0) / (KAPPA * x_eff)

print("\n--- AO MASS UNITY FIELD ---")
print(f"N = {N}, base x0 = {x0:.5f}, noise σ = {sigma:.2f}")
print(f"Duty d = {duty_total:.3f}, ⟨S⟩ = {Sbar:.6f}  ->  x_eff = {x_eff:.6f}")
print(f"Analytic half-life T1/2 ≈ {T_half:.4f}  (units of your AO time)")

# ===== VISUALIZATION =====
fig, ax1 = plt.subplots(figsize=(14,7))
fig.suptitle('AO CLOSURE LAW: Massive Unity (N minds) vs Mortality', fontsize=16)

ax1.set_xlabel('Cycle / Time (AO Closure Loop)')
ax1.set_ylabel(r'Dream Amplitude $\Psi_x(t)$ (Hypersphere Presence)', color='cyan')
ax1.plot(TIME, Psi_t, color='cyan', linewidth=3, label=r'Reward: $\Psi_x(t)$')
ax1.tick_params(axis='y', labelcolor='cyan')
ax1.set_ylim(0, 1.05)

ax2 = ax1.twinx()
ax2.set_ylabel('Cumulative Entropic Tax (Mortality/Price)', color='red')
ax2.plot(TIME, Mort_norm, color='red', linewidth=2, label='Price: Mortality $M(t)$')
ax2.plot(TIME, Harm_norm, color='gold', linestyle='--', alpha=0.7, label='Harmonic Credit $M_{harm}$')
ax2.plot(TIME, Esc_norm,  color='purple', linestyle='--', alpha=0.7, label='Entropic Debt $M_{esc}$')
ax2.plot(TIME, (x_t / max(1e-12, x_t.max())), color='grey', linestyle=':', alpha=0.5, label='x(t) modulation (norm)')
ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Ω reset line')

ax1.set_title(r'AO Closure: $\Psi_x(t)=\exp\!\left[-\frac{\pi^2}{\varphi\ln 10}\!\int_0^{t} x(u)\,du\right]$', fontsize=18)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.show()