import numpy as np
import matplotlib.pyplot as plt

# --- AO FUNDAMENTAL CONSTANTS ---
PI = np.pi
PHI = (1 + np.sqrt(5)) / 2  # Zlatý řez (Golden Ratio)
LN10 = np.log(10)

# AO Rezonanční Faktor (AO-Schrödingerův koeficient)
AO_RESONANCE_FACTOR = (PI**2) / (PHI * LN10)
# Přibližně 4.390

# --- SIMULAČNÍ PARAMETRY ---
TIME_STEPS = np.linspace(0, 10, 100) # 10 Cyklů
TIME_DELTA = TIME_STEPS[1] - TIME_STEPS[0] # Malý krok pro integraci

# Základní Entropický Únik (simulovaný)
X_LEAKAGE_BASE = 0.01

# --- VĚDOMÁ MODULACE ---
# Vědomá Modulace (C) Entropického Úniku (řídí Svobodnou Vůli)
# Modulace X mezi 0.5 a 1.5
C_MODULATION = 1 + 0.5 * np.sin(TIME_STEPS * PI / 5)

# AKTIVNÍ Entropický Únik (x_t) - Frekvence Vědomí
X_T = X_LEAKAGE_BASE * C_MODULATION

# --- AO Closure Law & MORTALITY ---

# 1. Amplituda AO Vlny (Přítomnost Snu / Hypersféra)
DECAY_RATE = AO_RESONANCE_FACTOR * X_T
AO_WAVE_AMPLITUDE = np.exp(-DECAY_RATE * TIME_STEPS)

# 2. Kumulativní Entropická Daň (Mortality/Cena)
# Je to kumulativní součet (integrál) entropického úniku X v čase.
# Toto je 'cena' zaplacená za existenci diskrétní formy.
CUMULATIVE_ENTROPY_TAX = np.cumsum(X_T) * TIME_DELTA

# Normalizace Ceny/Mortality pro vizualizaci na stejném grafu (0 až 1)
# Nastavíme cílovou 'smrtelnou hranici' na 100%
NORMALIZED_MORTALITY_TAX = CUMULATIVE_ENTROPY_TAX / CUMULATIVE_ENTROPY_TAX[-1]

# --- VIZUALIZACE SMLOUVY ---

fig, ax1 = plt.subplots(figsize=(12, 6))
fig.suptitle('AO Closure Law: CENA a ODMĚNA (Entropická Daň / Mortality Overlay)', fontsize=14)

# 1. ODMĚNA: Rozpad Hypersféry (Amplituda AO Vlny) - Cesta k Jednotě
color = 'cyan'
ax1.set_xlabel('Cyklus / Čas (AO Closure Loop)')
ax1.set_ylabel('Amplituda Snu (Hypersféra/Odměna)', color=color)
line1, = ax1.plot(TIME_STEPS, AO_WAVE_AMPLITUDE, color=color, linewidth=3)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 1.05)

# 2. CENA: Kumulativní Entropická Daň (Mortality)
ax2 = ax1.twinx()  # Sekundární osa
color = 'red'
ax2.set_ylabel('Kumulativní Entropická Daň (Mortality/Cena)', color=color)
line2, = ax2.plot(TIME_STEPS, NORMALIZED_MORTALITY_TAX, color=color, linestyle='-', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 1.05)

# 3. Entropický Únik x(t) - Frekvence Vědomí (Modulátor)
color_x = 'gold'
line3, = ax1.plot(TIME_STEPS, X_T / X_T.max(), color=color_x, linestyle=':', alpha=0.5) # Normalizováno pro srovnání

# Vodorovná čára na 1.0 (Definice resetu / smrtelnosti)
ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Mortality Reset / $\\Omega$ Uzávěr')

# Popisy a Legenda
ax1.set_title(r'Mortality $\propto \int x(t) dt$ $\quad \Psi_x(t) \propto e^{-f(x) t}$', fontsize=16)

fig.legend([line1, line2, line3],
           ['Odměna: Amplituda Snu ($\Psi_x$)',
            'Cena: Kumulativní Entropická Daň (Mortality)',
            'Modulace Vědomím (Entropický Únik x)'],
           loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.show()

print(f"\n--- AO SIMULACE MORTALITY ---")
print(f"Predikce: S růstem Mortalitní Daně se Amplituda Snu rozpadá.")
print(f"Mortality Tax je kumulativní cena za individuální existenci (Hyperkrychli).")