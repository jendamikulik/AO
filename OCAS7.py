#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chaos π-rational coherence probe for the logistic map.

Usage example:
  python chaos_pi_coherence.py --r 3.8 --eps 0.02 --steps 200000 --burn 5000 \
      --q_max 32 --compare_irrational --map_432 --fs 48000 --plot --record_some
"""

import math
import argparse
from collections import namedtuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAVE_PLOT = True
except Exception:
    HAVE_PLOT = False

Result = namedtuple("Result", "label omega mean std lyap trace_x trace_f")

# --------------------------- core dynamics & metrics ---------------------------

def logistic_step(x, r):
    # standard logistic map without forcing; derivative wrt x is r(1-2x)
    return r * x * (1 - x)

def lyap_increment(x, r):
    # |f'(x)| = |r(1-2x)|; to avoid log(0) clamp very small values
    val = abs(r * (1 - 2 * x))
    if val < 1e-15:
        val = 1e-15
    return math.log(val)

def run_perturbed_logistic(r=3.8, eps=0.02, steps=200000, burn=5000,
                           omega=math.pi, phi=0.0, seed=1234):
    """
    Iterate x_{t+1} = clip( r x_t (1-x_t) + eps * sin(omega*t + phi), 0, 1 )
    Return mean, std, lyapunov proxy after burn-in + the last trace (optional).
    """
    rng = np.random.default_rng(seed)
    x = rng.random()

    # burn-in
    for t in range(burn):
        drive = eps * math.sin(omega * t + phi)
        x = r * x * (1 - x) + drive
        if x < 0.0: x = 0.0
        if x > 1.0: x = 1.0

    # streaming mean / var (Welford) and Lyapunov sum
    n = 0
    mean = 0.0
    M2 = 0.0
    lyap_sum = 0.0

    trace_x = np.empty(0)  # filled only if needed by caller
    trace_f = np.empty(0)

    for t in range(steps):
        # stats on current x before next step (convention OK either way)
        n += 1
        delta = x - mean
        mean += delta / n
        M2 += delta * (x - mean)
        lyap_sum += lyap_increment(x, r)

        # step
        drive = eps * math.sin(omega * (t + burn) + phi)
        x = r * x * (1 - x) + drive
        if x < 0.0: x = 0.0
        if x > 1.0: x = 1.0

    std = math.sqrt(M2 / n) if n > 1 else float("nan")
    lyap = lyap_sum / n if n > 0 else float("nan")
    return mean, std, lyap

# --------------------------- frequency set builders ---------------------------

def reduced_fractions_upto(Qmax):
    """Yield reduced p/q with 1 <= q <= Qmax and 1 <= p < 2q (covers [0,2π))."""
    for q in range(1, Qmax + 1):
        for p in range(1, 2 * q):  # exclude 0 and 2q to avoid duplicates
            if math.gcd(p, q) == 1:
                yield p, q

def build_frequency_list(q_max=32, include_irr=False, include_432=False, fs=48000):
    """
    Returns list of (label, omega) pairs:
      - all reduced π * p/q with q ≤ q_max
      - optional irrational π*sqrt(2)
      - optional mapped_432 = 2π * 432 / fs
    """
    freq = []
    for p, q in reduced_fractions_upto(q_max):
        lbl = f"pi*{p}/{q}"
        omg = math.pi * (p / q)
        freq.append((lbl, omg))

    if include_irr:
        freq.append(("pi*sqrt(2)", math.pi * math.sqrt(2.0)))

    if include_432:
        # map 432 Hz onto [0, 2π) using discrete-time angular frequency 2π f / fs
        omg = 2.0 * math.pi * 432.0 / float(fs)
        freq.append(("mapped_432", omg))

    return freq

# --------------------------- run sweep & pretty print --------------------------

def run_sweep(r=3.8, eps=0.02, steps=200000, burn=5000,
              q_max=32, compare_irrational=False, map_432=False, fs=48000,
              seed=1234):
    freqs = build_frequency_list(q_max, include_irr=compare_irrational,
                                 include_432=map_432, fs=fs)
    results = []
    for label, omg in freqs:
        mu, sd, ly = run_perturbed_logistic(r=r, eps=eps, steps=steps,
                                            burn=burn, omega=omg, seed=seed)
        results.append(Result(label, omg, mu, sd, ly, None, None))
    return results

def print_table_header(r, eps):
    print(f"Perturbed logistic map  r={r}  eps={eps}")
    print(f"{'label':22} {'omega(rad)':>12} {'mean':>10} {'std':>10} {'lyap':>10}")

def print_result_row(res: Result):
    def fmt(x): return "   nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:10.4f}"
    print(f"{res.label:22} {res.omega:12.6f} {fmt(res.mean)} {fmt(res.std)} {fmt(res.lyap)}")

# --------------------------- optional plotting & recording ---------------------

def pick_interesting(results, k=6):
    """
    Pick a few representatives:
      - 2 with smallest lyap
      - 2 around median lyap
      - 2 with largest lyap
    """
    arr = sorted(results, key=lambda r: r.lyap)
    if len(arr) <= k:
        return arr
    lo = arr[:2]
    mid = [arr[len(arr)//2 - 1], arr[len(arr)//2]]
    hi = arr[-2:]
    # unique by label order preserved
    seen, out = set(), []
    for r in lo + mid + hi:
        if r.label not in seen:
            seen.add(r.label)
            out.append(r)
    return out

def record_some(results, r, eps, steps, burn, seed=1234, out_prefix="record"):
    """
    Re-run selected labels to capture short traces for plots: x_t and drive f_t.
    Saves PNGs if matplotlib is available; otherwise silently skips plotting.
    """
    if not HAVE_PLOT:
        return

    sel = pick_interesting(results, k=6)
    for res in sel:
        # Re-run to get traces (short window for visuals)
        Tplot = min(5000, steps)
        x = np.empty(Tplot, dtype=float)
        f = np.empty(Tplot, dtype=float)

        rng = np.random.default_rng(seed)
        xv = rng.random()

        # burn-in
        for t in range(burn):
            drive = eps * math.sin(res.omega * t)
            xv = r * xv * (1 - xv) + drive
            if xv < 0.0: xv = 0.0
            if xv > 1.0: xv = 1.0

        # collect
        for t in range(Tplot):
            x[t] = xv
            f[t] = eps * math.sin(res.omega * (t + burn))
            xv = r * xv * (1 - xv) + f[t]
            if xv < 0.0: xv = 0.0
            if xv > 1.0: xv = 1.0

        # plots
        fig1 = plt.figure(figsize=(8, 3))
        plt.plot(x)
        plt.title(f"x_t  —  {res.label}, ω={res.omega:.6f}")
        plt.xlabel("t"); plt.ylabel("x")
        plt.tight_layout()
        fig1.savefig(f"{out_prefix}_{res.label}_xt.png", dpi=120)
        plt.close(fig1)

        fig2 = plt.figure(figsize=(8, 3))
        plt.plot(f)
        plt.title(f"drive f_t=ε sin(ωt) — {res.label}")
        plt.xlabel("t"); plt.ylabel("f_t")
        plt.tight_layout()
        fig2.savefig(f"{out_prefix}_{res.label}_drive.png", dpi=120)
        plt.close(fig2)

def plot_histogram_panels(results, r, eps, steps, burn, seed=1234):
    """Small multiples: histograms for a subset of frequencies."""
    if not HAVE_PLOT:
        return
    sel = pick_interesting(results, k=6)
    n = len(sel)
    cols = 3
    rows = int(math.ceil(n / cols))
    fig = plt.figure(figsize=(4 * cols, 3 * rows))

    for idx, res in enumerate(sel, start=1):
        # regenerate a short trajectory to histogram
        Tplot = min(40000, steps)
        rng = np.random.default_rng(seed)
        x = rng.random()

        # burn-in
        for t in range(burn):
            drive = eps * math.sin(res.omega * t)
            x = r * x * (1 - x) + drive
            if x < 0.0: x = 0.0
            if x > 1.0: x = 1.0

        xs = np.empty(Tplot, dtype=float)
        for t in range(Tplot):
            xs[t] = x
            drive = eps * math.sin(res.omega * (t + burn))
            x = r * x * (1 - x) + drive
            if x < 0.0: x = 0.0
            if x > 1.0: x = 1.0

        ax = fig.add_subplot(rows, cols, idx)
        ax.hist(xs, bins=80, density=True)
        ax.set_title(f"{res.label}\nω={res.omega:.6f}, λ={res.lyap:.4f}")
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.show()

# ----------------------------------- CLI --------------------------------------

def main():
    ap = argparse.ArgumentParser(description="π-rational coherence probe on logistic map")
    ap.add_argument("--r", type=float, default=3.8, help="logistic parameter r")
    ap.add_argument("--eps", type=float, default=0.02, help="forcing amplitude ε")
    ap.add_argument("--steps", type=int, default=200000, help="iterations after burn-in")
    ap.add_argument("--burn", type=int, default=5000, help="burn-in iterations")
    ap.add_argument("--q_max", type=int, default=32, help="max denominator q for π*p/q")
    ap.add_argument("--compare_irrational", action="store_true",
                    help="also include ω = π*sqrt(2)")
    ap.add_argument("--map_432", action="store_true",
                    help="include ω mapped from 432 Hz: ω=2π*432/fs")
    ap.add_argument("--fs", type=int, default=48000, help="sampling rate for 432 mapping")
    ap.add_argument("--plot", action="store_true", help="make quick hist/timeseries plots")
    ap.add_argument("--record_some", action="store_true",
                    help="save a few PNGs of selected traces/drives")
    ap.add_argument("--seed", type=int, default=1234, help="PRNG seed")
    args = ap.parse_args()

    results = run_sweep(
        r=args.r, eps=args.eps, steps=args.steps, burn=args.burn,
        q_max=args.q_max, compare_irrational=args.compare_irrational,
        map_432=args.map_432, fs=args.fs, seed=args.seed
    )

    print_table_header(args.r, args.eps)
    for res in results:
        print_result_row(res)

    if args.record_some:
        record_some(results, r=args.r, eps=args.eps, steps=args.steps,
                    burn=args.burn, seed=args.seed, out_prefix="record")

    if args.plot:
        if not HAVE_PLOT:
            print("\n[plot] matplotlib not available; skipping plots.")
        else:
            plot_histogram_panels(results, r=args.r, eps=args.eps,
                                  steps=args.steps, burn=args.burn, seed=args.seed)

if __name__ == "__main__":
    main()
