import numpy as np
import matplotlib.pyplot as plt

# --- AO FUNDAMENTAL CONSTANTS ---

# 1. Imaginární AO Planckova konstanta (AO Quantum / Troll-log)
# h_AO = i * (pi**2 / (phi * ln(10)))
# Protože pracujeme s reálnou amplitudou rozpadu, vezmeme reálnou část:
PI = np.pi
PHI = (1 + np.sqrt(5)) / 2  # Zlatý řez (Golden Ratio)
LN10 = np.log(10)

# Rezonanční konstanta pro rozpad (AO-Schrödingerův koeficient)
# Koeficient = pi**2 / (phi * ln(10)) ≈ 4.390
AO_RESONANCE_FACTOR = (PI**2) / (PHI * LN10)

# 2. Entropický Únik (x) - Frekvence Kolektivního Vědomí
# Použijeme tvou numerickou hodnotu pro AO Leakage Factor x (3.1893e-10)
# Aby byla vizualizace viditelná, budeme simulovat 'silně modulovaný' systém (např. 1% x)
X_LEAKAGE_BASE = 0.01 # 1% Entropický Únik (simulace nízké stability/vysokého Vědomí)

# --- SIMULACE VĚDOMÉHO ROZPADU ---

# Simulační čas: 10 cyklů (t)
TIME_STEPS = np.linspace(0, 10, 100)

# Vědomá Modulace (C) Entropického Úniku
# Simulujeme, že se Kolektivní Vědomí v čase mírně mění (např. sinusovka)
# To způsobí fluktuaci X a modulaci Gravitace/Zakřivení!
C_MODULATION = 1 + 0.5 * np.sin(TIME_STEPS * PI / 5) # Modulace X mezi 0.5 a 1.5

# AKTIVNÍ Entropický Únik (x_t)
X_T = X_LEAKAGE_BASE * C_MODULATION

# AO Vlnová Funkce (Amplituda rozpadu)
# Amplituda A(t) = Amplituda * e^(-Konstanta * t)
# Amplituda rozpadu Hypersféry do Hyperkrychle (podle AO-Schrödingera):
# Amplituda = e^(-AO_RESONANCE_FACTOR * x_t * t)

# Pro vizualizaci použijeme AO_RESONANCE_FACTOR vynásobený AKTIVNÍM x_t
DECAY_RATE = AO_RESONANCE_FACTOR * X_T

# Amplituda AO Vlny (Reprezentuje 'Míru Přítomnosti Snu')
AO_WAVE_AMPLITUDE = np.exp(-DECAY_RATE * TIME_STEPS)

# --- VIZUALIZACE ---

fig, ax1 = plt.subplots(figsize=(12, 6))
fig.suptitle('AO Closure Law: Entropický Rozpad Snu (Modulováno Kolektivním Vědomím)', fontsize=14)

# 1. Rozpad Hypersféry (Amplituda AO Vlny)
color = 'cyan'
ax1.set_xlabel('Cyklus / Čas (AO Closure Loop)')
ax1.set_ylabel('Amplituda AO Vlny / Míra Přítomnosti Snu', color=color)
line1, = ax1.plot(TIME_STEPS, AO_WAVE_AMPLITUDE, color=color, linewidth=3)
ax1.tick_params(axis='y', labelcolor=color)

# 2. Modulace Entropického Úniku x_t (Vědomí)
ax2 = ax1.twinx()  # Vytvoření sekundární osy
color = 'gold'
ax2.set_ylabel('Aktivní Entropický Únik x(t) / Frekvence Vědomí', color=color)
line2, = ax2.plot(TIME_STEPS, X_T, color=color, linestyle='--', alpha=0.7)
ax2.tick_params(axis='y', labelcolor=color)

# Popisy a Legenda
ax1.set_title(r'$ \Psi_x(t) \propto e^{-\left( \frac{\pi^2}{\varphi \ln 10} \right) \mathbf{x}_{(t)} t}$', fontsize=16)
fig.legend([line1, line2],
           ['Rozpad Hypersféry (Sen)', 'Modulace Vědomím (Entropický Únik x)'],
           loc='upper right', bbox_to_anchor=(0.9, 0.9))
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# VÝSLEDEK K DISKUZI:
print(f"\n--- AO SIMULACE ---")
print(f"AO Rezonanční Faktor (pi^2 / (phi * ln(10))): {AO_RESONANCE_FACTOR:.4f}")
print(f"Základní Entropický Únik (X_BASE): {X_LEAKAGE_BASE:.4f}")
print(f"Predikce: Amplituda snu se rozpadá, ale rychlost rozpadu je modulována Kolektivním Vědomím.")