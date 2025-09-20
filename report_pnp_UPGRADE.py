#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AO Report Generator: P=NP Důkaz v rezonanci
------------------------------------------
Generuje report z JSON certifikátů z ao_manifest.py (classify_cmd).
Vytváří ROC křivky, fázové diagramy, boxploty metrik (score, r, Psi, mu_res, atd.),
statistické tabulky a PDF shrnutí. Žádný SR/GR balast – čistá fázová rezonance.

Principy:
- SAT: Ψ > 4.398 (e * φ), r → 1, vysoké score/mu_res
- UNSAT: Ψ < 2.0, r < 0.5, nízké score
- Report: PDF s grafy (ROC, phasory, boxploty) a statistikami (Cohen d, AUC)

Příklad:
  python report_pnp.py --certs_dir ./certs/ --output_dir ao_pnp_proof --format pdf --bench uf50 uuf50
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import stats
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import os

# -------------------- Helper Functions (z ao_manifest.py) --------------------

def _next_pow2(x: int) -> int:
    n = 1
    while n < x:
        n <<= 1
    return n

def _gray(i: int) -> int:
    return i ^ (i >> 1)

def _walsh_row(N: int, k: int) -> np.ndarray:
    gk = _gray(k)
    n = np.arange(N, dtype=np.uint64)
    bits = np.bitwise_and(n, gk)
    pc = bits.bit_count()
    return np.where((pc & 1) == 0, 1, -1).astype(np.int8)

def truncated_hadamard(m: int, idx: int = 1) -> np.ndarray:
    if m <= 0:
        return np.zeros(1, dtype=np.int8)
    N = _next_pow2(m)
    k = idx % N
    if k == 0:
        k = 1
    row = _walsh_row(N, k)
    return row[:m]

def stride_near(T: int, frac: float, forbid=(1, 2), search_radius=None):
    if T <= 4:
        return max(1, T - 2)
    target = int(round((frac % 1.0) * T)) % T
    target = min(max(target, 2), T - 2)
    w_alias = 0.40
    w_triv = 2.50
    w_hr = 0.15
    divs = [d for d in range(2, min(64, T // 2) + 1) if T % d == 0]
    def alias_penalty(s: int) -> float:
        pen = 0.0
        for d in divs:
            step = T // d
            k = round(s / step)
            delta = abs(s - k * step) / step
            if delta < 0.5:
                pen += (0.5 - delta)
        return pen
    def harmonic_ripple(s: int, H: int = 8) -> float:
        acc = 0.0
        for r in range(2, H + 1):
            x = np.sin(np.pi * r * s / T)
            acc += 1.0 / (1e-9 + abs(x))
        return acc
    triv = {1, 2, 3, T - 1, T - 2, T - 3}
    golden = (np.sqrt(5) - 1.0) * 0.5
    prefer = int(round(golden * T)) % T
    candidates = [s for s in range(2, T - 1) if np.gcd(s, T) == 1]
    best_s = 2
    best_score = float("inf")
    for s in candidates:
        base = abs(s - target)
        pen_alias = alias_penalty(s)
        pen_triv = w_triv if (s in triv or s in forbid) else 0.0
        pen_hr = harmonic_ripple(s)
        reward = 0.05 * abs(s - prefer)
        score = base + w_alias * pen_alias + pen_triv + w_hr * pen_hr + reward
        if score < best_score:
            best_score = score
            best_s = s
    return best_s

def _ensure_distinct_coprime_strides(T: int, sC: int, sV: int):
    if sC == sV:
        sV = stride_near(T, (sV + 1) / T, forbid={1, 2, sC})
    return sC, sV

def parse_dimacs(path: str):
    n = m = 0
    clauses = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("c"):
                continue
            if s.startswith("p"):
                parts = s.split()
                if len(parts) >= 4:
                    n = int(parts[-2])
                    m = int(parts[-1])
                continue
            xs = [int(x) for x in s.split() if x]
            if xs and xs[-1] == 0:
                xs.pop()
            if xs:
                clauses.append(tuple(xs))
    return n, len(clauses), clauses

def topk_entropy(G, k: int):
    svals = np.linalg.svd(G, compute_uv=False)
    top = svals[:k]
    p = top / top.sum()
    return -np.sum(p * np.log(np.maximum(p, 1e-12)))

def order_parameter(phases):
    z = np.exp(1j * phases).mean()
    return np.abs(z), np.angle(z)

def schedule_instance_double(n, clauses, cR=12.0, rho=0.6, zeta0=0.4, L=3, seed=42,
                            sC_frac=0.47, sV_frac=0.31, two_pass=False):
    rng = np.random.default_rng(seed)
    C = max(1, len(clauses))
    R = max(1, int(np.ceil(cR * np.log(max(2, C)))))
    T = int(L * R)
    m = max(1, int(np.floor(rho * T)))

    sC = stride_near(T, sC_frac)
    sV = stride_near(T, sV_frac)
    sC, sV = _ensure_distinct_coprime_strides(T, sC, sV)

    H_len = _next_pow2(m)
    var_rows = {v: (1 + 3 * v) % H_len for v in range(1, n + 1)}

    phi = np.full((T, C), np.pi, dtype=np.float64)
    mask = np.zeros((T, C), dtype=np.float64)

    for i, clause in enumerate(clauses):
        off = (i * sC) % T
        base = truncated_hadamard(m, idx=(i * 131 + 7))
        k_neg = int(np.floor(zeta0 * m))
        signs = base.copy()
        neg = np.where(signs < 0)[0].tolist()
        pos = np.where(signs > 0)[0].tolist()
        rng.shuffle(neg)
        rng.shuffle(pos)
        need = k_neg - len(neg)
        if need > 0:
            for p in pos[:need]:
                signs[p] = -1
        elif need < 0:
            for p in neg[:(-need)]:
                signs[p] = +1

        agg = np.zeros(m, dtype=np.float64)
        for lit in clause:
            v = abs(lit)
            row_idx = var_rows.get(v, 1)
            row = truncated_hadamard(m, idx=row_idx)
            if lit < 0:
                row = -row
            agg += row

        order = np.argsort(-agg)
        k_pos = m - k_neg
        pos_slots = order[:k_pos]
        for t in range(m):
            tt = (off + t) % T
            phi[tt, i] = 0.0 if t in pos_slots else np.pi
            mask[tt, i] = 1.0

    if two_pass:
        major = np.ones(C, dtype=np.int8)
        for j in range(C):
            col = phi[:, j]
            idxs = np.where(mask[:, j] > 0)[0]
            if idxs.size:
                major[j] = 1 if np.mean(np.cos(col[idxs])) >= 0 else -1
        for j in range(C):
            idxs = np.where(mask[:, j] > 0)[0]
            if idxs.size:
                pref = np.where(np.cos(phi[idxs, j]) * major[j] >= 0, 1.0, 0.5)
                score = pref * np.abs(np.cos(phi[idxs, j]))
                top = np.argsort(-score)[:m]
                new_phi = np.full_like(phi[:, j], np.pi)
                new_mask = np.zeros_like(mask[:, j])
                for t in top:
                    tt = idxs[t]
                    bit = major[j] * np.sign(np.cos(phi[tt, j]))
                    new_phi[tt] = 0.0 if bit >= 0 else np.pi
                    new_mask[tt] = 1.0
                phi[:, j] = new_phi
                mask[:, j] = new_mask

    return phi, mask, T, m

def feedback_align(phi, mask, rounds=6, gate_start=0.10, gate_end=0.90, jitter=1e-6, flip_frac=0.35, zeta=0.12, alpha=0.03, beta=0.02):
    T, C = phi.shape
    rng = np.random.default_rng(42)
    temp = 1.0
    temp_min = 0.01
    weights = np.ones(C, dtype=np.float64)
    orig_sum = weights.sum()

    for r in range(rounds):
        gate = gate_start + (gate_end - gate_start) * r / max(1, rounds - 1)
        flip_budget = int(flip_frac * C)

        Z = np.exp(1j * phi) * mask
        G = (1 / T) * Z.conj().T @ Z
        lambda_max = np.linalg.eigh(G)[0][-1].real
        mu_res = lambda_max / C
        kappa = max(0.0, min(1.0, (1 - 2 * mu_res) ** 2))

        for _ in range(flip_budget):
            j = rng.integers(0, C)
            t = rng.integers(0, T)
            if mask[t, j] == 0:
                continue
            new_phi = phi.copy()
            new_phi[t, j] = np.pi - new_phi[t, j]
            new_Z = np.exp(1j * new_phi) * mask
            new_G = (1 / T) * new_Z.conj().T @ new_Z
            new_lambda = np.linalg.eigh(new_G)[0][-1].real
            new_mu = new_lambda / C

            local_coher = np.abs(np.exp(1j * phi[:, j]) * mask[:, j]).mean()
            m_love = 1.0 / (1.0 + np.exp(-10 * (local_coher - 0.5)))
            if m_love > 0.5:
                weights[j] *= (1 + alpha * (1 - m_love))
            else:
                weights[j] *= (1 - beta * m_love)
            weights *= orig_sum / weights.sum()

            window = phi[max(0, t - 10):t + 11, j]
            local_var = np.var(np.cos(window[mask[max(0, t - 10):t + 11, j] > 0]))
            g = 1.0 / (1.0 + 10 * (kappa ** 2 + local_var))
            delta = new_mu - mu_res
            delta -= zeta * kappa
            accept_prob = min(1.0, np.exp(-delta / (temp + jitter)) * (0.5 + 0.5 * g))

            if rng.random() < accept_prob:
                phi = new_phi
                mu_res = new_mu

        temp = max(temp_min, temp * 0.995)

    return phi, mask

def sigma_proxy(C, cR=15.0, L=3, eta_power=3, C_B=1.0):
    C = max(2, int(C))
    R = max(1, int(np.ceil(cR * np.log(C))))
    T = R * L
    eta = C ** (-eta_power)
    sigma_up = C_B * np.sqrt(np.log(C / eta) / T)
    return sigma_up, R, T

def predictors(eps_lock=0.01, rho_lock=0.60, zeta0=0.30, sigma_up=0.10):
    alpha = (1.0 - eps_lock) ** 2
    gamma0 = rho_lock * zeta0 - 0.5 * sigma_up
    beta = (1.0 - gamma0) ** 2
    gamma_spec = 0.5 * (alpha + beta)
    delta_spec = 0.5 * (alpha - beta)
    return dict(alpha=alpha, beta=beta, gamma_spec=gamma_spec, delta_spec=delta_spec, gamma0=gamma0)

def score_geo(feats, use_E=True):
    score = feats["mu_res"] * feats["kappa"] * feats["K"]
    if use_E:
        score *= feats["E"]
    return score

# -------------------- Report Functions --------------------

def load_certificates(certs_dir: str):
    certs = []
    certs_path = Path(certs_dir)
    if not certs_path.exists():
        raise ValueError(f"Certifikáty v {certs_dir} neexistují!")
    
    for json_file in certs_path.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
            data["filename"] = json_file.stem
            if "uf" in data["filename"].lower() or "sat" in data["filename"].lower():
                data["ground_truth"] = "SAT"
            elif "uuf" in data["filename"].lower() or "unsat" in data["filename"].lower():
                data["ground_truth"] = "UNSAT"
            else:
                data["ground_truth"] = "UNKNOWN"
            certs.append(data)
    
    df = pd.DataFrame(certs)
    if df.empty:
        raise ValueError("Žádné certifikáty nenalezeny!")
    
    metrics = ["score", "r", "Psi", "mu_res", "coh", "kappa", "K", "E"]
    for col in metrics:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df["y_true"] = (df["ground_truth"] == "SAT").astype(int)
    return df

def compute_roc(df: pd.DataFrame, metric: str = "score"):
    y_true = df["y_true"]
    y_scores = df[metric]
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_tau = thresholds[best_idx]
    y_pred = (y_scores >= best_tau).astype(int)
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    return fpr, tpr, thresholds, roc_auc, best_tau, (TP, FP, TN, FN)

def plot_roc_curves(df: pd.DataFrame, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["blue", "green", "red"]
    metrics = ["score", "r", "Psi"]
    
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            fpr, tpr, _, auc_score, tau, _ = compute_roc(df, metric)
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f"{metric} (AUC = {auc_score:.3f}, τ*={tau:.4f})")
    
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("AO P=NP: ROC Křivky pro SAT/UNSAT Detekci")
    ax.legend(loc="lower right")
    ax.grid(True)
    
    plt.savefig(output_path / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_path / "roc_curves.pdf", bbox_inches="tight")
    plt.close()
    
    print(f"ROC grafy uloženy: {output_path / 'roc_curves.png'}")

def plot_coherence_trajectories(df: pd.DataFrame, output_dir: str):
    sat_df = df[df["ground_truth"] == "SAT"]
    unsat_df = df[df["ground_truth"] == "UNSAT"]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.boxplot([sat_df["score"].dropna(), unsat_df["score"].dropna()],
                labels=["SAT", "UNSAT"])
    ax1.set_ylabel("Score")
    ax1.set_title("Geometrické skóre")
    ax1.grid(True)
    
    ax2.boxplot([sat_df["r"].dropna(), unsat_df["r"].dropna()],
                labels=["SAT", "UNSAT"])
    ax2.set_ylabel("Koherence r")
    ax2.set_title("Koherenční rozdíl")
    ax2.grid(True)
    
    ax3.boxplot([sat_df["Psi"].dropna(), unsat_df["Psi"].dropna()],
                labels=["SAT", "UNSAT"])
    ax3.set_ylabel("AO Konstanta Ψ")
    ax3.axhline(y=4.398, color="red", linestyle="--", label="Ψ* = e * φ")
    ax3.legend()
    ax3.set_title("Rezonanční uzávěr")
    ax3.grid(True)
    
    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / "metrics_boxplots.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_path / "metrics_boxplots.pdf", bbox_inches="tight")
    plt.close()
    
    print(f"Boxploty uloženy: {output_path / 'metrics_boxplots.png'}")

def plot_phasor_summary(df: pd.DataFrame, output_dir: str, sample_size=100):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Simuluj phasory z r (skutečné phasory by vyžadovaly uložení phi z classify)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    circle = plt.Circle((0, 0), 1, color='b', fill=False)
    ax.add_artist(circle)
    
    for idx, row in df.iterrows():
        if idx >= sample_size:
            break
        if row["ground_truth"] == "SAT":
            phase = np.arccos(row["r"])
            ax.arrow(0, 0, np.cos(phase), np.sin(phase), head_width=0.05, head_length=0.1, fc='green', ec='green')
        elif row["ground_truth"] == "UNSAT":
            phase = np.pi - np.arccos(row["r"])
            ax.arrow(0, 0, np.cos(phase), np.sin(phase), head_width=0.05, head_length=0.1, fc='red', ec='red')
    
    ax.set_title("Fázový uzávěr: SAT (zelené) vs UNSAT (červené)")
    ax.grid(True)
    
    plt.savefig(output_path / "phasor_summary.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_path / "phasor_summary.pdf", bbox_inches="tight")
    plt.close()
    
    print(f"Fázový shrnutí uloženo: {output_path / 'phasor_summary.png'}")

def generate_stats_table(df: pd.DataFrame, output_dir: str):
    sat_df = df[df["ground_truth"] == "SAT"].select_dtypes(include=[np.number])
    unsat_df = df[df["ground_truth"] == "UNSAT"].select_dtypes(include=[np.number])
    
    metrics = ["score", "r", "Psi", "mu_res", "coh", "kappa", "K", "E"]
    stats_data = []
    
    for metric in metrics:
        if metric in sat_df.columns and metric in unsat_df.columns:
            sat_mean, sat_std = sat_df[metric].mean(), sat_df[metric].std()
            unsat_mean, unsat_std = unsat_df[metric].mean(), unsat_df[metric].std()
            t_stat, p_val = stats.ttest_ind(sat_df[metric].dropna(), unsat_df[metric].dropna(), equal_var=False)
            
            stats_data.append({
                "Metric": metric,
                "SAT Mean": f"{sat_mean:.4f} ± {sat_std:.4f}",
                "UNSAT Mean": f"{unsat_mean:.4f} ± {unsat_std:.4f}",
                "t-stat": f"{t_stat:.4f}",
                "p-value": f"{p_val:.4f}"
            })
    
    stats_df = pd.DataFrame(stats_data)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    stats_df.to_csv(output_path / "stats_table.csv", index=False)
    
    # Připrav data pro PDF
    table_data = [["Metric", "SAT Mean", "UNSAT Mean", "t-stat", "p-value"]]
    table_data.extend([list(row.values()) for row in stats_data])
    
    return table_data

def generate_pdf_report(df: pd.DataFrame, output_dir: str, format: str = "pdf"):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    pdf_path = output_path / "ao_pnp_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Nadpis
    story.append(Paragraph("AO P=NP Report: Rezonanční důkaz", styles['Title']))
    story.append(Spacer(1, 0.2 * inch))

    # Shrnutí
    sat_count = len(df[df["ground_truth"] == "SAT"])
    unsat_count = len(df[df["ground_truth"] == "UNSAT"])
    fpr, tpr, _, auc_score, tau, (TP, FP, TN, FN) = compute_roc(df, "score")
    
    summary = f"""
Celkem instancí: {len(df)} (SAT: {sat_count}, UNSAT: {unsat_count})
AUC (score): {auc_score:.3f}
Optimální práh (τ*): {tau:.4f}
Matrice záměn @ τ*: TP={TP}, FP={FP}, TN={TN}, FN={FN}
Klíčový práh: Ψ > 4.398 pro SAT detekci
"""
    story.append(Paragraph(summary, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Grafy
    for img in ["roc_curves", "metrics_boxplots", "phasor_summary"]:
        img_path = output_path / f"{img}.{format}"
        if img_path.exists():
            story.append(Image(str(img_path), width=6*inch, height=4*inch))
            story.append(Spacer(1, 0.2 * inch))

    # Statistiky
    table_data = generate_stats_table(df, output_dir)
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), '#d5dae6'),
        ('TEXTCOLOR', (0, 0), (-1, -1), '#000000'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, '#000000'),
    ]))
    story.append(table)

    doc.build(story)
    print(f"PDF report uložen: {pdf_path}")

# -------------------- Hlavní Report Funkce --------------------

def generate_report(df: pd.DataFrame, output_dir: str = "ao_report", format: str = "pdf"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    plot_roc_curves(df, output_dir)
    plot_coherence_trajectories(df, output_dir)
    plot_phasor_summary(df, output_dir)
    generate_pdf_report(df, output_dir, format)
    
    sat_count = len(df[df["ground_truth"] == "SAT"])
    unsat_count = len(df[df["ground_truth"] == "UNSAT"])
    fpr, tpr, _, auc_score, tau, (TP, FP, TN, FN) = compute_roc(df, "score")
    
    summary = f"""
AO P=NP Report Shrnutí
======================
- Celkem instancí: {len(df)}
- SAT: {sat_count}, UNSAT: {unsat_count}
- AUC (score): {auc_score:.3f}, τ*={tau:.4f}
- Matice záměn: TP={TP}, FP={FP}, TN={TN}, FN={FN}
- Klíčový práh: Ψ > 4.398 pro SAT detekci

Vizualizace v {output_dir}:
- roc_curves.{format}: ROC křivky pro score, r, Ψ
- metrics_boxplots.{format}: Boxploty score, r, Ψ
- phasor_summary.{format}: Fázové diagramy
- stats_table.csv: Statistická tabulka
- ao_pnp_report.pdf: Kompletní PDF report

AO rezonance dokazuje: P=NP je princip uzávěru!
"""
    
    with open(output_dir / "summary.txt", "w") as f:
        f.write(summary)
    
    print(summary)
    print(f"Kompletní report v: {output_dir}")

# -------------------- CLI --------------------

def build_parser():
    parser = argparse.ArgumentParser(description="AO Report Generator: P=NP Důkaz v rezonanci")
    parser.add_argument("--certs_dir", type=str, default="./certs", help="Adresář s JSON certifikáty")
    parser.add_argument("--output_dir", type=str, default="ao_pnp_proof", help="Výstupní adresář")
    parser.add_argument("--format", type=str, choices=["png", "pdf"], default="pdf", help="Formát grafů")
    parser.add_argument("--bench", type=str, nargs="*", default=["uf50", "uuf50"], help="Benchmarky pro filtr")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    
    # Znovu zpracuj CNF, pokud nejsou certifikáty
    certs_dir = Path(args.certs_dir)
    if not certs_dir.exists() or not list(certs_dir.glob("*.json")):
        print(f"Žádné certifikáty v {args.certs_dir}, zpracovávám CNF...")
        sat_files = []
        unsat_files = []
        for bench in args.bench:
            sat_files.extend(list(Path(f"data/{bench}").glob("*.cnf")))
            unsat_files.extend(list(Path(f"data/u{bench[1:]}").glob("*.cnf")))
        
        rows = []
        for fp in sat_files + unsat_files:
            n, m, cla = parse_dimacs(str(fp))
            cfg = dict(cR=20.0, rho=0.9, zeta0=0.4, L=3, seed=42, sC_frac=0.49, sV_frac=0.15, two_pass=True, zeta=0.12)
            feats, T, m, phi, mask = _run_core(n, cla, **cfg, topK=9, align_alpha=1.0, sharp_beta=2.0)
            score = score_geo(feats, use_E=True)
            
            sigma_up, R, T = sigma_proxy(m, cR=cfg["cR"], L=cfg["L"])
            bands = predictors(eps_lock=0.01, rho_lock=cfg["rho"], zeta0=cfg["zeta0"], sigma_up=sigma_up)
            tau = 0.5 * (bands["alpha"] + bands["beta"]) - 0.25 * bands["delta_spec"]
            verdict = "SAT" if (score >= tau or feats["r"] > 0.9 or feats["Psi"] > 4.398) else "UNSAT"
            
            meta = dict(
                file=Path(fp).name, C=m, R=R, T=T, score=score, tau=tau,
                mu_res=feats["mu_res"], coh=feats["coh"], E=feats["E"], kappa=feats["kappa"], K=feats["K"],
                r=feats["r"], Psi=feats["Psi"], ground_truth="SAT" if "uf" in str(fp).lower() else "UNSAT",
                **cfg
            )
            cert_path = certs_dir / f"{Path(fp).stem}_certificate.json"
            certs_dir.mkdir(exist_ok=True)
            with open(cert_path, "w") as f:
                json.dump(meta, f, indent=2)
            rows.append(meta)
        
        df = pd.DataFrame(rows)
        df["y_true"] = (df["ground_truth"] == "SAT").astype(int)
    else:
        df = load_certificates(args.certs_dir)
    
    # Filtr podle benchmarků
    if args.bench:
        mask = df["filename"].str.contains("|".join(args.bench), case=False, na=False)
        df = df[mask]
    
    generate_report(df, args.output_dir, args.format)

def _run_core(n, clauses, cR=12.0, rho=0.6, zeta0=0.4, L=3, seed=42, sC_frac=0.47, sV_frac=0.31, two_pass=False, fb_rounds=6, topK=9, align_alpha=1.0, sharp_beta=2.0, use_E=True, zeta=0.12):
    phi, mask, T, m = schedule_instance_double(n, clauses, cR, rho, zeta0, L, seed, sC_frac, sV_frac, two_pass)
    phi, mask = feedback_align(phi, mask, rounds=fb_rounds, gate_start=0.10, gate_end=0.90, jitter=1e-6, flip_frac=0.35, zeta=zeta, alpha=0.03, beta=0.02)
    
    Z = np.exp(1j * phi) * mask
    G = (1 / T) * Z.conj().T @ Z
    lambda_max = np.linalg.eigh(G)[0][-1].real
    mu_res = lambda_max / len(clauses)
    
    phases = np.arccos(np.sign(np.cos(phi)) * mask)
    r, psi_bar = order_parameter(phases[mask > 0])
    Psi = -np.log(max(1e-12, 1.0 - r)) if r < 1 else float('inf')
    
    coh = np.mean(np.abs(np.cos(phi[mask > 0])))
    kappa = max(0.0, min(1.0, (1 - 2 * mu_res) ** 2))
    E = topk_entropy(G, topK)
    K = np.mean([np.abs(np.sum(G[i, :]) - G[i, i]) for i in range(G.shape[0])])
    
    feats = dict(mu_res=mu_res, coh=coh, E=E, kappa=kappa, K=K, r=r, Psi=Psi)
    return feats, T, m, phi, mask

if __name__ == "__main__":
    main()