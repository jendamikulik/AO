# AO Framework: Absolute Oscillation for Resonance and Coherence

This repository implements the Absolute Operator (AO) framework for simulating resonant constants Ψ ≈ 4.398 (e * φ), SAT/UNSAT dynamics in oscillator networks, and physical demos (Sagnac, MMX, GPS, Hafele-Keating). It demonstrates how frustration (UNSAT-like) leads to higher Ψ accumulation without full coherence, bridging chaos and harmony.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run all demos: `python ao_allinone.py`
   - Outputs: Ψ/r convergence prints, figures for SAT/UNSAT oscillators, Sagnac/MMX/GPS results.
3. Test: `pytest tests/`

## Key Features
- **Oscillator Network**: Kuramoto-style with AO gates (love/noise/zone-mix ζ). SAT-like: high r, stable Ψ; UNSAT-like: high Ψ, mid r (frustrated resonance).
- **Ψ Computation**: Ψ = (kappa / D_est) * r, with Bessel validation (r_pred = I1(Ψ)/I0(Ψ)).
- **Physical Demos**: Sagnac effect, MMX null result, GPS relativity correction, Hafele-Keating time dilation.
- **Philosophy**: AO replaces SR/GR fluff with energy transfer and phase closure. See docs in code.

## Examples
- CNF files in `examples/cnf/` for SAT/UNSAT testing (loadable via future extensions).

License: MIT
