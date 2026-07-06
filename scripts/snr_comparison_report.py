#!/usr/bin/env python
"""Generate an HTML comparison report of SNR metrics: Minipigs vs Monkeys.

Loads pre-computed channel and session tables from both species and generates
comparative plots and an HTML report highlighting differences in SNR distributions.

Usage::

    uv run python scripts/snr_comparison_report.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp

OUTPUT_DIR = Path("outputs/snr_comparison")
MINIPIG_DIR = Path("outputs/snr_analysis")
MONKEY_DIR = Path("outputs/snr_analysis_monkeys")

SPECIES_COLORS = {"Minipigs": "#3498db", "Monkeys": "#e74c3c"}
SNR_THRESHOLD = 0.5


def load_data():
    """Load channel and session tables for both species."""
    mp_ch = pd.read_csv(MINIPIG_DIR / "channel_quality_table.csv")
    mp_ses = pd.read_csv(MINIPIG_DIR / "session_summary_table.csv")
    mk_ch = pd.read_csv(MONKEY_DIR / "channel_quality_table.csv")
    mk_ses = pd.read_csv(MONKEY_DIR / "session_summary_table.csv")

    mp_ch["species"] = "Minipigs"
    mk_ch["species"] = "Monkeys"
    mp_ses["species"] = "Minipigs"
    mk_ses["species"] = "Monkeys"

    return mp_ch, mp_ses, mk_ch, mk_ses


def plot_snr_histogram_comparison(mp_ch, mk_ch, output_path):
    """Overlaid histograms of broadband ERP SNR for both species."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    mp_snr = mp_ch["broadband_snr"].values
    mk_snr = mk_ch["broadband_snr"].values

    # Panel 1: Full range (capped at 99th percentile)
    ax = axes[0]
    max_val = max(np.percentile(mp_snr, 99), np.percentile(mk_snr, 99))
    bins = np.linspace(0, max_val * 1.1, 60)

    ax.hist(
        mp_snr,
        bins=bins,
        alpha=0.6,
        color=SPECIES_COLORS["Minipigs"],
        edgecolor="black",
        linewidth=0.3,
        label=f"Minipigs (n={len(mp_snr)})",
        density=True,
    )
    ax.hist(
        mk_snr,
        bins=bins,
        alpha=0.6,
        color=SPECIES_COLORS["Monkeys"],
        edgecolor="black",
        linewidth=0.3,
        label=f"Monkeys (n={len(mk_snr)})",
        density=True,
    )
    ax.axvline(
        SNR_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold = {SNR_THRESHOLD}",
    )
    ax.set_xlabel("Broadband Evoked SNR")
    ax.set_ylabel("Density")
    ax.set_title("ERP SNR Distribution (full range)")
    ax.legend(fontsize=9)

    # Panel 2: Zoomed into low range (where most values sit)
    ax2 = axes[1]
    bins2 = np.linspace(0, 0.15, 50)
    ax2.hist(
        mp_snr,
        bins=bins2,
        alpha=0.6,
        color=SPECIES_COLORS["Minipigs"],
        edgecolor="black",
        linewidth=0.3,
        label="Minipigs",
        density=True,
    )
    ax2.hist(
        mk_snr,
        bins=bins2,
        alpha=0.6,
        color=SPECIES_COLORS["Monkeys"],
        edgecolor="black",
        linewidth=0.3,
        label="Monkeys",
        density=True,
    )
    ax2.set_xlabel("Broadband Evoked SNR")
    ax2.set_ylabel("Density")
    ax2.set_title("ERP SNR Distribution (zoomed: 0–0.15)")
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Evoked-Potential SNR: Minipigs vs Monkeys",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_power_ratio_comparison(mp_ch, mk_ch, output_path):
    """Comparison of power ratio SNR between species."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    mp_pr = mp_ch["broadband_power_snr"].values
    mk_pr = mk_ch["broadband_power_snr"].values

    # Panel 1: Histograms
    ax = axes[0]
    max_val = min(np.percentile(np.concatenate([mp_pr, mk_pr]), 95), 5.0)
    bins = np.linspace(0, max_val, 50)
    ax.hist(
        mp_pr,
        bins=bins,
        alpha=0.6,
        color=SPECIES_COLORS["Minipigs"],
        edgecolor="black",
        linewidth=0.3,
        label="Minipigs",
        density=True,
    )
    ax.hist(
        mk_pr,
        bins=bins,
        alpha=0.6,
        color=SPECIES_COLORS["Monkeys"],
        edgecolor="black",
        linewidth=0.3,
        label="Monkeys",
        density=True,
    )
    ax.axvline(
        1.0, color="black", linestyle="--", linewidth=1.5, label="Ratio = 1.0"
    )
    ax.set_xlabel("Power Ratio SNR")
    ax.set_ylabel("Density")
    ax.set_title("Power Ratio Distribution")
    ax.legend(fontsize=8)

    # Panel 2: Box plots
    ax2 = axes[1]
    data_to_plot = [
        mp_pr[mp_pr < np.percentile(mp_pr, 95)],
        mk_pr[mk_pr < np.percentile(mk_pr, 95)],
    ]
    bp = ax2.boxplot(
        data_to_plot,
        tick_labels=["Minipigs", "Monkeys"],
        patch_artist=True,
        widths=0.6,
    )
    bp["boxes"][0].set_facecolor(SPECIES_COLORS["Minipigs"])
    bp["boxes"][1].set_facecolor(SPECIES_COLORS["Monkeys"])
    for box in bp["boxes"]:
        box.set_alpha(0.6)
    ax2.axhline(1.0, color="black", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Power Ratio SNR")
    ax2.set_title("Power Ratio (< 95th percentile)")

    # Panel 3: Fraction above 1.0 per session
    ax3 = axes[2]
    mp_frac = (
        mp_ch
        .groupby("session_id")["broadband_power_snr"]
        .apply(lambda x: (x > 1.0).mean())
        .values
    )
    mk_frac = (
        mk_ch
        .groupby("session_id")["broadband_power_snr"]
        .apply(lambda x: (x > 1.0).mean())
        .values
    )

    bp3 = ax3.boxplot(
        [mp_frac, mk_frac],
        tick_labels=["Minipigs", "Monkeys"],
        patch_artist=True,
        widths=0.6,
    )
    bp3["boxes"][0].set_facecolor(SPECIES_COLORS["Minipigs"])
    bp3["boxes"][1].set_facecolor(SPECIES_COLORS["Monkeys"])
    for box in bp3["boxes"]:
        box.set_alpha(0.6)
    ax3.set_ylabel("Fraction of channels with Power Ratio > 1.0")
    ax3.set_title("Per-Session: Channels above threshold")
    ax3.set_ylim(-0.05, 1.05)

    fig.suptitle(
        "Power Ratio SNR: Minipigs vs Monkeys", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_session_level_comparison(mp_ses, mk_ses, output_path):
    """Session-level power ratio comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Session mean power ratio, sorted
    ax = axes[0]
    mp_vals = (
        mp_ses["mean_broadband_power_snr"].sort_values(ascending=False).values
    )
    mk_vals = (
        mk_ses["mean_broadband_power_snr"].sort_values(ascending=False).values
    )

    ax.bar(
        range(len(mp_vals)),
        mp_vals,
        alpha=0.7,
        color=SPECIES_COLORS["Minipigs"],
        label="Minipigs",
        width=0.8,
    )
    offset = len(mp_vals) + 2
    ax.bar(
        range(offset, offset + len(mk_vals)),
        mk_vals,
        alpha=0.7,
        color=SPECIES_COLORS["Monkeys"],
        label="Monkeys",
        width=0.8,
    )
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Session (sorted by power ratio)")
    ax.set_ylabel("Mean Power Ratio SNR")
    ax.set_title("Per-Session Mean Power Ratio")
    ax.legend()

    # Panel 2: Distribution of session-mean power ratios
    ax2 = axes[1]
    bins = np.linspace(0, max(mp_vals.max(), mk_vals.max()) * 1.05, 25)
    ax2.hist(
        mp_vals,
        bins=bins,
        alpha=0.6,
        color=SPECIES_COLORS["Minipigs"],
        edgecolor="black",
        linewidth=0.3,
        label="Minipigs",
    )
    ax2.hist(
        mk_vals,
        bins=bins,
        alpha=0.6,
        color=SPECIES_COLORS["Monkeys"],
        edgecolor="black",
        linewidth=0.3,
        label="Monkeys",
    )
    ax2.axvline(1.0, color="black", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Session Mean Power Ratio SNR")
    ax2.set_ylabel("Number of Sessions")
    ax2.set_title("Distribution of Session-Level Power Ratio")
    ax2.legend()

    fig.suptitle("Session-Level Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_responsive_ratio_comparison(mp_ch, mk_ch, output_path):
    """Responsive ratio comparison between species."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    mp_rr = mp_ch["broadband_resp_ratio"].values
    mk_rr = mk_ch["broadband_resp_ratio"].values

    # Panel 1: Histograms
    ax = axes[0]
    bins = np.linspace(0.3, 0.9, 40)
    ax.hist(
        mp_rr,
        bins=bins,
        alpha=0.6,
        color=SPECIES_COLORS["Minipigs"],
        edgecolor="black",
        linewidth=0.3,
        label="Minipigs",
        density=True,
    )
    ax.hist(
        mk_rr,
        bins=bins,
        alpha=0.6,
        color=SPECIES_COLORS["Monkeys"],
        edgecolor="black",
        linewidth=0.3,
        label="Monkeys",
        density=True,
    )
    ax.axvline(
        0.5, color="black", linestyle="--", linewidth=1, label="Chance (50%)"
    )
    ax.set_xlabel("Responsive Ratio")
    ax.set_ylabel("Density")
    ax.set_title("Trial-by-Trial Responsive Ratio")
    ax.legend(fontsize=9)

    # Panel 2: Box plot
    ax2 = axes[1]
    bp = ax2.boxplot(
        [mp_rr, mk_rr],
        tick_labels=["Minipigs", "Monkeys"],
        patch_artist=True,
        widths=0.6,
    )
    bp["boxes"][0].set_facecolor(SPECIES_COLORS["Minipigs"])
    bp["boxes"][1].set_facecolor(SPECIES_COLORS["Monkeys"])
    for box in bp["boxes"]:
        box.set_alpha(0.6)
    ax2.axhline(0.5, color="black", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Responsive Ratio")
    ax2.set_title("Trial-Level Reliability")

    fig.suptitle(
        "Responsive Ratio: Minipigs vs Monkeys", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_erp_vs_power_scatter(mp_ch, mk_ch, output_path):
    """Scatter plot of ERP SNR vs Power Ratio for both species."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for species, df, color in [
        ("Minipigs", mp_ch, SPECIES_COLORS["Minipigs"]),
        ("Monkeys", mk_ch, SPECIES_COLORS["Monkeys"]),
    ]:
        evoked = df["broadband_snr"].values
        power = df["broadband_power_snr"].values
        # Clip for visualization
        mask = (power < np.percentile(power, 98)) & (
            evoked < np.percentile(evoked, 98)
        )
        ax.scatter(
            evoked[mask],
            power[mask],
            alpha=0.3,
            s=15,
            color=color,
            edgecolors="none",
            label=species,
        )

    ax.axhline(1.0, color="black", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(0.5, color="black", linestyle=":", alpha=0.4, linewidth=1)
    ax.set_xlabel("Evoked SNR (ERP, phase-locked)")
    ax.set_ylabel("Power Ratio SNR (phase-insensitive)")
    ax.set_title("ERP SNR vs Power Ratio: Both Species")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_stats(mp_ch, mk_ch, mp_ses, mk_ses):
    """Compute summary statistics for the comparison."""
    stats = {}

    # Dataset overview
    stats["mp_n_channels"] = len(mp_ch)
    stats["mk_n_channels"] = len(mk_ch)
    stats["mp_n_sessions"] = len(mp_ses)
    stats["mk_n_sessions"] = len(mk_ses)
    stats["mp_n_subjects"] = (
        mp_ch["session_id"].apply(lambda x: x.split("_")[0]).nunique()
    )
    stats["mk_n_subjects"] = (
        mk_ch["session_id"].apply(lambda x: x.split("_")[0]).nunique()
    )
    stats["mp_channels_per_session"] = (
        f"{mp_ses['total_channels'].min()}–{mp_ses['total_channels'].max()}"
    )
    stats["mk_channels_per_session"] = (
        f"{mk_ses['total_channels'].min()}–{mk_ses['total_channels'].max()}"
    )

    # ERP SNR
    stats["mp_erp_mean"] = mp_ch["broadband_snr"].mean()
    stats["mp_erp_median"] = mp_ch["broadband_snr"].median()
    stats["mk_erp_mean"] = mk_ch["broadband_snr"].mean()
    stats["mk_erp_median"] = mk_ch["broadband_snr"].median()
    stats["mp_erp_active"] = (mp_ch["status"] == "Active").sum()
    stats["mk_erp_active"] = (mk_ch["status"] == "Active").sum()
    stats["mp_erp_active_pct"] = 100 * stats["mp_erp_active"] / len(mp_ch)
    stats["mk_erp_active_pct"] = 100 * stats["mk_erp_active"] / len(mk_ch)

    # ERP SNR percentiles
    for pct in [25, 50, 75, 90, 95, 99]:
        stats[f"mp_erp_p{pct}"] = np.percentile(mp_ch["broadband_snr"], pct)
        stats[f"mk_erp_p{pct}"] = np.percentile(mk_ch["broadband_snr"], pct)

    # Power Ratio SNR
    stats["mp_power_mean"] = mp_ch["broadband_power_snr"].mean()
    stats["mp_power_median"] = mp_ch["broadband_power_snr"].median()
    stats["mk_power_mean"] = mk_ch["broadband_power_snr"].mean()
    stats["mk_power_median"] = mk_ch["broadband_power_snr"].median()
    stats["mp_power_above1"] = (mp_ch["broadband_power_snr"] > 1.0).sum()
    stats["mk_power_above1"] = (mk_ch["broadband_power_snr"] > 1.0).sum()
    stats["mp_power_above1_pct"] = 100 * stats["mp_power_above1"] / len(mp_ch)
    stats["mk_power_above1_pct"] = 100 * stats["mk_power_above1"] / len(mk_ch)

    # Power Ratio percentiles
    for pct in [25, 50, 75, 90, 95, 99]:
        stats[f"mp_power_p{pct}"] = np.percentile(
            mp_ch["broadband_power_snr"], pct
        )
        stats[f"mk_power_p{pct}"] = np.percentile(
            mk_ch["broadband_power_snr"], pct
        )

    # Session-level power ratio
    stats["mp_ses_power_mean"] = mp_ses["mean_broadband_power_snr"].mean()
    stats["mk_ses_power_mean"] = mk_ses["mean_broadband_power_snr"].mean()
    stats["mp_ses_power_above1"] = (
        mp_ses["mean_broadband_power_snr"] > 1.0
    ).sum()
    stats["mk_ses_power_above1"] = (
        mk_ses["mean_broadband_power_snr"] > 1.0
    ).sum()
    stats["mp_ses_power_above1_pct"] = (
        100 * stats["mp_ses_power_above1"] / len(mp_ses)
    )
    stats["mk_ses_power_above1_pct"] = (
        100 * stats["mk_ses_power_above1"] / len(mk_ses)
    )
    stats["mp_ses_power_std"] = mp_ses["mean_broadband_power_snr"].std()
    stats["mk_ses_power_std"] = mk_ses["mean_broadband_power_snr"].std()

    # Responsive ratio
    stats["mp_resp_mean"] = mp_ch["broadband_resp_ratio"].mean()
    stats["mk_resp_mean"] = mk_ch["broadband_resp_ratio"].mean()
    stats["mp_resp_median"] = mp_ch["broadband_resp_ratio"].median()
    stats["mk_resp_median"] = mk_ch["broadband_resp_ratio"].median()

    # Statistical tests
    u_erp, p_erp = mannwhitneyu(
        mp_ch["broadband_snr"], mk_ch["broadband_snr"], alternative="two-sided"
    )
    stats["mwu_erp_u"] = u_erp
    stats["mwu_erp_p"] = p_erp

    u_power, p_power = mannwhitneyu(
        mp_ch["broadband_power_snr"],
        mk_ch["broadband_power_snr"],
        alternative="two-sided",
    )
    stats["mwu_power_u"] = u_power
    stats["mwu_power_p"] = p_power

    ks_erp, p_ks_erp = ks_2samp(mp_ch["broadband_snr"], mk_ch["broadband_snr"])
    stats["ks_erp_stat"] = ks_erp
    stats["ks_erp_p"] = p_ks_erp

    ks_power, p_ks_power = ks_2samp(
        mp_ch["broadband_power_snr"], mk_ch["broadband_power_snr"]
    )
    stats["ks_power_stat"] = ks_power
    stats["ks_power_p"] = p_ks_power

    # Responsive ratio test
    u_resp, p_resp = mannwhitneyu(
        mp_ch["broadband_resp_ratio"],
        mk_ch["broadband_resp_ratio"],
        alternative="two-sided",
    )
    stats["mwu_resp_u"] = u_resp
    stats["mwu_resp_p"] = p_resp

    return stats


def generate_html_report(stats, output_path):
    """Generate the full HTML comparison report."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SNR Comparison Report &mdash; Minipigs vs Monkeys</title>
    <style>
        :root {{
            --primary: #2c3e50;
            --accent: #3498db;
            --accent2: #e74c3c;
            --bg: #f8f9fa;
            --card-bg: #ffffff;
            --border: #dee2e6;
            --text: #212529;
            --text-light: #6c757d;
            --success: #27ae60;
            --danger: #e74c3c;
            --warning: #f39c12;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.7;
        }}
        header {{
            background: linear-gradient(135deg, #2c3e50, #8e44ad);
            color: white;
            padding: 3rem 2rem;
            text-align: center;
        }}
        header h1 {{ font-size: 2.2rem; font-weight: 700; margin-bottom: 0.5rem; }}
        header p {{ font-size: 1.1rem; opacity: 0.85; max-width: 700px; margin: 0 auto; }}
        .container {{ max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }}
        .card {{
            background: var(--card-bg);
            border-radius: 10px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
            padding: 2rem;
            margin-bottom: 2rem;
        }}
        h2 {{
            color: var(--primary);
            font-size: 1.6rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid var(--accent);
        }}
        h3 {{ color: var(--primary); font-size: 1.25rem; margin: 1.5rem 0 0.75rem; }}
        p {{ margin-bottom: 0.75rem; }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }}
        .stat-box {{
            background: var(--bg);
            border-radius: 8px;
            padding: 1.2rem;
            text-align: center;
            border: 1px solid var(--border);
        }}
        .stat-box .value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--accent);
            display: block;
        }}
        .stat-box .label {{
            font-size: 0.85rem;
            color: var(--text-light);
            margin-top: 0.25rem;
        }}
        .stat-box.monkey .value {{ color: var(--accent2); }}
        .stat-box.success .value {{ color: var(--success); }}
        .stat-box.warning .value {{ color: var(--warning); }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            margin: 1rem 0;
        }}
        th, td {{
            padding: 0.6rem 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{ background: var(--primary); color: white; font-weight: 600; }}
        tr:hover {{ background: #f1f3f5; }}
        .figure {{ text-align: center; margin: 1.5rem 0; }}
        .figure img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .figure figcaption {{
            color: var(--text-light);
            font-size: 0.9rem;
            margin-top: 0.5rem;
            font-style: italic;
        }}
        .finding {{
            background: #eaf4fd;
            border-left: 4px solid var(--accent);
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            border-radius: 0 6px 6px 0;
        }}
        .finding.warning {{ background: #fef9e7; border-left-color: var(--warning); }}
        .finding.success {{ background: #e8f8f0; border-left-color: var(--success); }}
        .finding.danger {{ background: #fdecea; border-left-color: var(--danger); }}
        .finding strong {{ color: var(--primary); }}
        .toc {{
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
        }}
        .toc h3 {{ margin-top: 0; }}
        .toc ol {{ padding-left: 1.5rem; }}
        .toc li {{ margin: 0.4rem 0; }}
        .toc a {{ color: var(--accent); text-decoration: none; }}
        .toc a:hover {{ text-decoration: underline; }}
        .species-tag {{
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .species-minipig {{ background: #dbeafe; color: #1d4ed8; }}
        .species-monkey {{ background: #fce4ec; color: #c62828; }}
        .comparison-table {{ overflow-x: auto; }}
        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-light);
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <header>
        <h1>SNR Comparison: Minipigs vs Monkeys</h1>
        <p>Comparative analysis of iEEG signal quality across species using evoked-potential
        and power-ratio SNR metrics</p>
    </header>

    <div class="container">

        <div class="toc">
            <h3>Table of Contents</h3>
            <ol>
                <li><a href="#overview">Overview &amp; Hypothesis</a></li>
                <li><a href="#datasets">Dataset Summary</a></li>
                <li><a href="#erp-snr">Evoked-Potential (ERP) SNR Comparison</a></li>
                <li><a href="#power-ratio">Power Ratio SNR Comparison</a></li>
                <li><a href="#responsive-ratio">Trial-by-Trial Reliability Comparison</a></li>
                <li><a href="#session-level">Session-Level Analysis</a></li>
                <li><a href="#statistical-tests">Statistical Tests</a></li>
                <li><a href="#conclusions">Conclusions</a></li>
            </ol>
        </div>

        <!-- SECTION 1: OVERVIEW -->
        <div class="card" id="overview">
            <h2>1. Overview &amp; Hypothesis</h2>

            <h3>Motivation</h3>
            <p>
                This report compares iEEG signal quality between minipig and monkey recordings
                from the Neurosoft acoustic stimulation paradigm. Both species underwent the same
                experimental protocol (tone-pip stimulation with 0.5&thinsp;s on/off epochs) and
                were processed through identical pipelines.
            </p>

            <h3>Hypothesis</h3>
            <div class="finding">
                <strong>Primary Hypothesis:</strong> Monkey recordings are expected to show higher
                SNR than minipigs, as the monkey data appears qualitatively cleaner during visual
                inspection. If confirmed, this would suggest that the electrode&ndash;cortex interface
                in monkeys provides better signal transduction for auditory-evoked responses.
            </div>

            <h3>Methodology</h3>
            <p>
                Two complementary SNR metrics are compared:
            </p>
            <ol>
                <li><strong>Evoked-Potential (ERP) SNR:</strong> Measures phase-locked, time-locked
                responses. Computed as Var(trial-averaged stimulus waveform) / mean(single-trial
                rest variance). Sensitive to consistent neural deflections.</li>
                <li><strong>Power Ratio SNR:</strong> Measures total power increase during stimulus
                vs rest, regardless of phase consistency. Computed as mean(stimulus variance) /
                mean(rest variance). Sensitive to any energy modulation.</li>
            </ol>
            <p>
                Both metrics use 0.5&thinsp;s windows with broadband filtering (1&ndash;300&thinsp;Hz,
                50&thinsp;Hz notch). The ERP metric requires phase-locked responses to accumulate
                across trials, while the power ratio captures both phase-locked and non-phase-locked
                (induced) activity.
            </p>
        </div>

        <!-- SECTION 2: DATASETS -->
        <div class="card" id="datasets">
            <h2>2. Dataset Summary</h2>

            <div class="stat-grid">
                <div class="stat-box">
                    <span class="value">{stats["mp_n_subjects"]}</span>
                    <span class="label">Minipig Subjects</span>
                </div>
                <div class="stat-box monkey">
                    <span class="value">{stats["mk_n_subjects"]}</span>
                    <span class="label">Monkey Subjects</span>
                </div>
                <div class="stat-box">
                    <span class="value">{stats["mp_n_sessions"]}</span>
                    <span class="label">Minipig Sessions</span>
                </div>
                <div class="stat-box monkey">
                    <span class="value">{stats["mk_n_sessions"]}</span>
                    <span class="label">Monkey Sessions</span>
                </div>
            </div>

            <table>
                <tr>
                    <th>Property</th>
                    <th><span class="species-tag species-minipig">Minipigs</span></th>
                    <th><span class="species-tag species-monkey">Monkeys</span></th>
                </tr>
                <tr>
                    <td>Subjects</td>
                    <td>{stats["mp_n_subjects"]}</td>
                    <td>{stats["mk_n_subjects"]}</td>
                </tr>
                <tr>
                    <td>Sessions</td>
                    <td>{stats["mp_n_sessions"]}</td>
                    <td>{stats["mk_n_sessions"]}</td>
                </tr>
                <tr>
                    <td>Total ECoG Channels</td>
                    <td>{stats["mp_n_channels"]}</td>
                    <td>{stats["mk_n_channels"]}</td>
                </tr>
                <tr>
                    <td>Channels per Session</td>
                    <td>{stats["mp_channels_per_session"]}</td>
                    <td>{stats["mk_channels_per_session"]}</td>
                </tr>
            </table>

            <p>
                The monkey dataset has a more uniform electrode configuration (29 channels per session),
                whereas minipig recordings vary substantially (2&ndash;32 channels). The monkey dataset
                is dominated by sub-01 (16 sessions), with 5 other subjects contributing 1&ndash;5
                sessions each.
            </p>
        </div>

        <!-- SECTION 3: ERP SNR -->
        <div class="card" id="erp-snr">
            <h2>3. Evoked-Potential (ERP) SNR Comparison</h2>
            <p>
                The ERP SNR measures how much consistent, phase-locked stimulus response a channel
                shows relative to its noise floor. Values &gt;&thinsp;0.5 indicate reliably
                detectable auditory responses.
            </p>

            <h3>Summary Statistics</h3>
            <div class="stat-grid">
                <div class="stat-box">
                    <span class="value">{stats["mp_erp_median"]:.4f}</span>
                    <span class="label">Minipig Median ERP SNR</span>
                </div>
                <div class="stat-box monkey">
                    <span class="value">{stats["mk_erp_median"]:.4f}</span>
                    <span class="label">Monkey Median ERP SNR</span>
                </div>
                <div class="stat-box">
                    <span class="value">{stats["mp_erp_active"]}/{
        stats["mp_n_channels"]
    }</span>
                    <span class="label">Minipig Active ({
        stats["mp_erp_active_pct"]:.1f}%)</span>
                </div>
                <div class="stat-box monkey">
                    <span class="value">{stats["mk_erp_active"]}/{
        stats["mk_n_channels"]
    }</span>
                    <span class="label">Monkey Active ({
        stats["mk_erp_active_pct"]:.1f}%)</span>
                </div>
            </div>

            <h3>Percentile Distribution</h3>
            <table>
                <tr>
                    <th>Percentile</th>
                    <th>25th</th>
                    <th>50th</th>
                    <th>75th</th>
                    <th>90th</th>
                    <th>95th</th>
                    <th>99th</th>
                </tr>
                <tr>
                    <td><span class="species-tag species-minipig">Minipigs</span></td>
                    <td>{stats["mp_erp_p25"]:.4f}</td>
                    <td>{stats["mp_erp_p50"]:.4f}</td>
                    <td>{stats["mp_erp_p75"]:.4f}</td>
                    <td>{stats["mp_erp_p90"]:.4f}</td>
                    <td>{stats["mp_erp_p95"]:.4f}</td>
                    <td>{stats["mp_erp_p99"]:.4f}</td>
                </tr>
                <tr>
                    <td><span class="species-tag species-monkey">Monkeys</span></td>
                    <td>{stats["mk_erp_p25"]:.4f}</td>
                    <td>{stats["mk_erp_p50"]:.4f}</td>
                    <td>{stats["mk_erp_p75"]:.4f}</td>
                    <td>{stats["mk_erp_p90"]:.4f}</td>
                    <td>{stats["mk_erp_p95"]:.4f}</td>
                    <td>{stats["mk_erp_p99"]:.4f}</td>
                </tr>
            </table>

            <div class="figure">
                <img src="erp_snr_comparison.png" alt="ERP SNR histogram comparison">
                <figcaption>Figure 1: Distribution of broadband evoked-potential SNR for both species.
                Left: full range; Right: zoomed into the dominant low-SNR region.</figcaption>
            </div>

            <div class="finding {
        "success"
        if stats["mk_erp_median"] > stats["mp_erp_median"]
        else "warning"
    }">
                <strong>Key Finding &mdash; ERP SNR:</strong>
                {
        "Monkey channels show higher median ERP SNR than minipigs"
        if stats["mk_erp_median"] > stats["mp_erp_median"]
        else "Monkey channels show LOWER median ERP SNR than minipigs"
    } ({stats["mk_erp_median"]:.4f} vs {stats["mp_erp_median"]:.4f}).
                {
        "However, neither species has many channels exceeding the 0.5 threshold."
        if stats["mk_erp_active_pct"] < 5
        else ""
    }
                The minipig dataset has {
        stats["mp_erp_active"]
    } active channels ({stats["mp_erp_active_pct"]:.1f}%)
                while monkeys have {stats["mk_erp_active"]} ({
        stats["mk_erp_active_pct"]:.1f}%).
                This suggests that {
        "both species" if stats["mk_erp_active"] < 5 else "minipigs primarily"
    } lack strong phase-locked auditory evoked potentials
                at the ECoG level in these recordings.
            </div>
        </div>

        <!-- SECTION 4: POWER RATIO -->
        <div class="card" id="power-ratio">
            <h2>4. Power Ratio SNR Comparison</h2>
            <p>
                The power ratio SNR measures whether stimulus periods have higher total energy than
                rest periods, regardless of phase consistency. A ratio &gt;&thinsp;1.0 means the
                channel shows increased power during stimulation. This metric captures both evoked
                (phase-locked) and induced (non-phase-locked) neural responses.
            </p>

            <h3>Summary Statistics</h3>
            <div class="stat-grid">
                <div class="stat-box">
                    <span class="value">{stats["mp_power_mean"]:.3f}</span>
                    <span class="label">Minipig Mean Power Ratio</span>
                </div>
                <div class="stat-box monkey">
                    <span class="value">{stats["mk_power_mean"]:.3f}</span>
                    <span class="label">Monkey Mean Power Ratio</span>
                </div>
                <div class="stat-box">
                    <span class="value">{
        stats["mp_power_above1_pct"]:.1f}%</span>
                    <span class="label">Minipig Channels &gt; 1.0</span>
                </div>
                <div class="stat-box monkey">
                    <span class="value">{
        stats["mk_power_above1_pct"]:.1f}%</span>
                    <span class="label">Monkey Channels &gt; 1.0</span>
                </div>
            </div>

            <h3>Percentile Distribution</h3>
            <table>
                <tr>
                    <th>Percentile</th>
                    <th>25th</th>
                    <th>50th</th>
                    <th>75th</th>
                    <th>90th</th>
                    <th>95th</th>
                    <th>99th</th>
                </tr>
                <tr>
                    <td><span class="species-tag species-minipig">Minipigs</span></td>
                    <td>{stats["mp_power_p25"]:.4f}</td>
                    <td>{stats["mp_power_p50"]:.4f}</td>
                    <td>{stats["mp_power_p75"]:.4f}</td>
                    <td>{stats["mp_power_p90"]:.4f}</td>
                    <td>{stats["mp_power_p95"]:.4f}</td>
                    <td>{stats["mp_power_p99"]:.4f}</td>
                </tr>
                <tr>
                    <td><span class="species-tag species-monkey">Monkeys</span></td>
                    <td>{stats["mk_power_p25"]:.4f}</td>
                    <td>{stats["mk_power_p50"]:.4f}</td>
                    <td>{stats["mk_power_p75"]:.4f}</td>
                    <td>{stats["mk_power_p90"]:.4f}</td>
                    <td>{stats["mk_power_p95"]:.4f}</td>
                    <td>{stats["mk_power_p99"]:.4f}</td>
                </tr>
            </table>

            <div class="figure">
                <img src="power_ratio_comparison.png" alt="Power ratio SNR comparison">
                <figcaption>Figure 2: Power Ratio SNR comparison. Left: histogram overlay; Center: box plots
                (capped at 95th percentile); Right: per-session fraction of channels exceeding 1.0.</figcaption>
            </div>

            <div class="figure">
                <img src="erp_vs_power_scatter.png" alt="ERP vs Power Ratio scatter">
                <figcaption>Figure 3: Relationship between ERP SNR (x-axis) and Power Ratio SNR (y-axis)
                for both species. Points above the horizontal line show increased stimulus power.</figcaption>
            </div>

            <div class="finding {
        "success"
        if stats["mk_power_mean"] > stats["mp_power_mean"]
        else "danger"
    }">
                <strong>Key Finding &mdash; Power Ratio:</strong>
                {
        "Monkey channels show substantially higher mean power ratio than minipigs"
        if stats["mk_power_mean"] > stats["mp_power_mean"]
        else "Monkey channels show lower mean power ratio than minipigs"
    } ({stats["mk_power_mean"]:.3f} vs {stats["mp_power_mean"]:.3f}).
                {
        stats[
            "mk_power_above1_pct"
        ]:.1f}% of monkey channels show power increase during
                stimulation (ratio &gt; 1.0), compared to {
        stats["mp_power_above1_pct"]:.1f}% of
                minipig channels. {
        "This strongly supports the hypothesis that monkey recordings capture more stimulus-driven neural activity."
        if stats["mk_power_mean"] > stats["mp_power_mean"] * 1.2
        else ""
    }
            </div>
        </div>

        <!-- SECTION 5: RESPONSIVE RATIO -->
        <div class="card" id="responsive-ratio">
            <h2>5. Trial-by-Trial Reliability Comparison</h2>
            <p>
                The responsive ratio measures the fraction of individual trials where the stimulus
                window has higher variance than the rest window. By chance, this should be ~50%.
                Values well above 50% indicate consistent trial-level stimulus responses.
            </p>

            <div class="stat-grid">
                <div class="stat-box">
                    <span class="value">{
        stats["mp_resp_mean"] * 100:.1f}%</span>
                    <span class="label">Minipig Mean Resp. Ratio</span>
                </div>
                <div class="stat-box monkey">
                    <span class="value">{
        stats["mk_resp_mean"] * 100:.1f}%</span>
                    <span class="label">Monkey Mean Resp. Ratio</span>
                </div>
                <div class="stat-box">
                    <span class="value">{
        stats["mp_resp_median"] * 100:.1f}%</span>
                    <span class="label">Minipig Median Resp. Ratio</span>
                </div>
                <div class="stat-box monkey">
                    <span class="value">{
        stats["mk_resp_median"] * 100:.1f}%</span>
                    <span class="label">Monkey Median Resp. Ratio</span>
                </div>
            </div>

            <div class="figure">
                <img src="responsive_ratio_comparison.png" alt="Responsive ratio comparison">
                <figcaption>Figure 4: Trial-by-trial responsive ratio comparison. Left: density histograms;
                Right: box plots. The dashed line at 0.5 represents chance level.</figcaption>
            </div>

            <div class="finding {
        "success"
        if stats["mk_resp_mean"] > stats["mp_resp_mean"]
        else "warning"
    }">
                <strong>Key Finding &mdash; Responsive Ratio:</strong>
                {
        "Monkeys show higher mean responsive ratio than minipigs"
        if stats["mk_resp_mean"] > stats["mp_resp_mean"]
        else "Monkeys show similar or lower responsive ratio compared to minipigs"
    } ({stats["mk_resp_mean"] * 100:.1f}% vs {
        stats["mp_resp_mean"] * 100:.1f}%).
                {
        "This indicates that monkey channels more consistently respond to stimulation on a trial-by-trial basis."
        if stats["mk_resp_mean"] > stats["mp_resp_mean"] + 0.02
        else "Both species show similar trial-level consistency."
    }
            </div>
        </div>

        <!-- SECTION 6: SESSION LEVEL -->
        <div class="card" id="session-level">
            <h2>6. Session-Level Analysis</h2>
            <p>
                Session-level comparisons reveal whether signal quality differences are driven by
                a few exceptional sessions or represent a consistent species-level effect.
            </p>

            <div class="stat-grid">
                <div class="stat-box">
                    <span class="value">{stats["mp_ses_power_above1"]}/{
        stats["mp_n_sessions"]
    }</span>
                    <span class="label">Minipig Sessions &gt; 1.0 ({
        stats["mp_ses_power_above1_pct"]:.0f}%)</span>
                </div>
                <div class="stat-box monkey">
                    <span class="value">{stats["mk_ses_power_above1"]}/{
        stats["mk_n_sessions"]
    }</span>
                    <span class="label">Monkey Sessions &gt; 1.0 ({
        stats["mk_ses_power_above1_pct"]:.0f}%)</span>
                </div>
            </div>

            <div class="figure">
                <img src="session_level_comparison.png" alt="Session-level comparison">
                <figcaption>Figure 5: Session-level power ratio. Left: bar chart of per-session means (sorted);
                Right: distribution of session-level means.</figcaption>
            </div>

            <div class="finding">
                <strong>Session-Level Summary:</strong>
                {
        stats[
            "mk_ses_power_above1_pct"
        ]:.0f}% of monkey sessions have mean power ratio above 1.0
                (indicating net stimulus-driven power increase), compared to
                {stats["mp_ses_power_above1_pct"]:.0f}% of minipig sessions.
                The monkey dataset shows {
        "higher variability"
        if stats["mk_ses_power_std"] > stats["mp_ses_power_std"]
        else "comparable variability"
    } across sessions (std = {stats["mk_ses_power_std"]:.3f} vs {
        stats["mp_ses_power_std"]:.3f}).
            </div>
        </div>

        <!-- SECTION 7: STATISTICAL TESTS -->
        <div class="card" id="statistical-tests">
            <h2>7. Statistical Tests</h2>
            <p>
                Formal statistical comparisons between species distributions. Due to unequal sample
                sizes and non-normal distributions, non-parametric tests are used.
            </p>

            <table>
                <tr>
                    <th>Test</th>
                    <th>Metric</th>
                    <th>Statistic</th>
                    <th>p-value</th>
                    <th>Interpretation</th>
                </tr>
                <tr>
                    <td>Mann-Whitney U</td>
                    <td>ERP SNR</td>
                    <td>U = {stats["mwu_erp_u"]:.0f}</td>
                    <td>{stats["mwu_erp_p"]:.2e}</td>
                    <td>{
        "Significant" if stats["mwu_erp_p"] < 0.05 else "Not significant"
    }</td>
                </tr>
                <tr>
                    <td>Mann-Whitney U</td>
                    <td>Power Ratio SNR</td>
                    <td>U = {stats["mwu_power_u"]:.0f}</td>
                    <td>{stats["mwu_power_p"]:.2e}</td>
                    <td>{
        "Significant" if stats["mwu_power_p"] < 0.05 else "Not significant"
    }</td>
                </tr>
                <tr>
                    <td>Mann-Whitney U</td>
                    <td>Responsive Ratio</td>
                    <td>U = {stats["mwu_resp_u"]:.0f}</td>
                    <td>{stats["mwu_resp_p"]:.2e}</td>
                    <td>{
        "Significant" if stats["mwu_resp_p"] < 0.05 else "Not significant"
    }</td>
                </tr>
                <tr>
                    <td>KS 2-sample</td>
                    <td>ERP SNR</td>
                    <td>D = {stats["ks_erp_stat"]:.4f}</td>
                    <td>{stats["ks_erp_p"]:.2e}</td>
                    <td>{
        "Distributions differ" if stats["ks_erp_p"] < 0.05 else "No difference"
    }</td>
                </tr>
                <tr>
                    <td>KS 2-sample</td>
                    <td>Power Ratio SNR</td>
                    <td>D = {stats["ks_power_stat"]:.4f}</td>
                    <td>{stats["ks_power_p"]:.2e}</td>
                    <td>{
        "Distributions differ"
        if stats["ks_power_p"] < 0.05
        else "No difference"
    }</td>
                </tr>
            </table>
        </div>

        <!-- SECTION 8: CONCLUSIONS -->
        <div class="card" id="conclusions">
            <h2>8. Conclusions</h2>

            <h3>Hypothesis Evaluation</h3>
"""

    # Dynamic conclusions based on results
    erp_higher = stats["mk_erp_median"] > stats["mp_erp_median"]
    power_higher = stats["mk_power_mean"] > stats["mp_power_mean"]
    power_much_higher = stats["mk_power_mean"] > stats["mp_power_mean"] * 1.5

    if power_higher and not erp_higher:
        conclusion_class = "warning"
        conclusion_text = """
            The hypothesis is <strong>partially supported</strong>. Monkey recordings show
            substantially higher <em>power ratio</em> SNR, indicating greater stimulus-driven
            energy modulation. However, <em>evoked-potential</em> (ERP) SNR is not higher in
            monkeys, suggesting that the monkey auditory responses may be less phase-locked
            than minipigs (or phase-lock differently across trials).<br><br>
            This dissociation implies that monkey neural activity changes robustly during
            stimulation (higher power), but the precise temporal pattern varies across trials
            (lower phase-locking). This could reflect a different neural coding strategy or
            differences in electrode placement relative to primary auditory cortex.
"""
    elif power_higher and erp_higher:
        conclusion_class = "success"
        conclusion_text = """
            The hypothesis is <strong>strongly supported</strong>. Monkey recordings show higher
            SNR on both metrics: evoked-potential SNR (phase-locked responses) and power ratio
            SNR (total energy modulation). This confirms that monkey iEEG provides cleaner
            auditory signals than minipig recordings in this paradigm.
"""
    else:
        conclusion_class = "danger"
        conclusion_text = """
            The hypothesis is <strong>not supported</strong>. Monkey recordings do not show
            consistently higher SNR than minipigs on either metric. The species may have
            comparable signal quality, or differences may be due to electrode placement
            rather than inherent species-level factors.
"""

    html += f"""
            <div class="finding {conclusion_class}">
                {conclusion_text}
            </div>

            <h3>Key Takeaways</h3>
            <ol>
                <li><strong>ERP SNR:</strong> {
        "Monkeys" if erp_higher else "Minipigs"
    } show
                {
        "higher" if erp_higher else "higher"
    } median evoked-potential SNR
                ({max(stats["mk_erp_median"], stats["mp_erp_median"]):.4f} vs
                {min(stats["mk_erp_median"], stats["mp_erp_median"]):.4f}), but
                {
        "both species"
        if stats["mk_erp_active"] < 10 and stats["mp_erp_active"] < 20
        else "one species"
    } have very few channels exceeding the 0.5 Active threshold.</li>

                <li><strong>Power Ratio:</strong> {
        "Monkey" if power_higher else "Minipig"
    } channels
                show {
        "dramatically" if power_much_higher else "moderately"
    } higher power ratio
                (mean {
        max(stats["mk_power_mean"], stats["mp_power_mean"]):.3f} vs
                {min(stats["mk_power_mean"], stats["mp_power_mean"]):.3f}),
                indicating {
        "stronger" if power_higher else "weaker"
    } total energy modulation
                during stimulation.</li>

                <li><strong>Responsive Ratio:</strong> {
        "Monkeys"
        if stats["mk_resp_mean"] > stats["mp_resp_mean"]
        else "Minipigs"
    }
                show higher trial-level consistency
                ({
        max(stats["mk_resp_mean"], stats["mp_resp_mean"]) * 100:.1f}% vs
                {
        min(stats["mk_resp_mean"], stats["mp_resp_mean"]) * 100:.1f}%).</li>

                <li><strong>ERP vs Power Dissociation:</strong> {
        "The large gap between power ratio and ERP SNR in monkeys suggests that their auditory responses involve substantial non-phase-locked (induced) activity, rather than stereotyped evoked potentials."
        if power_higher and not erp_higher
        else "Both metrics tell a consistent story across species."
    }</li>
            </ol>

            <h3>Implications for Decoding</h3>
            <p>
                {
        "The higher power ratio in monkey recordings suggests that power-based features (band power, spectral features) may be more informative for auditory decoding in monkeys than time-domain ERP features. For minipigs, the few Active channels with high ERP SNR suggest that phase-locked features from specific channels remain valuable."
        if power_higher
        else "Both species show similar signal characteristics, suggesting comparable decoding approaches should work."
    }
            </p>
        </div>

    </div>

    <footer>
        <p>Generated from pre-computed SNR analysis tables &mdash;
        Minipigs: {stats["mp_n_sessions"]} sessions, {
        stats["mp_n_channels"]
    } channels |
        Monkeys: {stats["mk_n_sessions"]} sessions, {
        stats["mk_n_channels"]
    } channels</p>
    </footer>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    mp_ch, mp_ses, mk_ch, mk_ses = load_data()

    print(f"  Minipigs: {len(mp_ch)} channels, {len(mp_ses)} sessions")
    print(f"  Monkeys:  {len(mk_ch)} channels, {len(mk_ses)} sessions")

    print("\nComputing statistics...")
    stats = compute_stats(mp_ch, mk_ch, mp_ses, mk_ses)

    print("\nGenerating plots...")
    plot_snr_histogram_comparison(
        mp_ch, mk_ch, OUTPUT_DIR / "erp_snr_comparison.png"
    )
    plot_power_ratio_comparison(
        mp_ch, mk_ch, OUTPUT_DIR / "power_ratio_comparison.png"
    )
    plot_session_level_comparison(
        mp_ses, mk_ses, OUTPUT_DIR / "session_level_comparison.png"
    )
    plot_responsive_ratio_comparison(
        mp_ch, mk_ch, OUTPUT_DIR / "responsive_ratio_comparison.png"
    )
    plot_erp_vs_power_scatter(
        mp_ch, mk_ch, OUTPUT_DIR / "erp_vs_power_scatter.png"
    )

    print("\nGenerating HTML report...")
    generate_html_report(stats, OUTPUT_DIR / "snr_comparison_report.html")

    print(
        f"\nDone! Report saved to: {OUTPUT_DIR / 'snr_comparison_report.html'}"
    )

    # Print key comparison stats
    print("\n" + "=" * 60)
    print("KEY COMPARISON RESULTS")
    print("=" * 60)
    print("\n  ERP SNR:")
    print(f"    Minipigs median: {stats['mp_erp_median']:.6f}")
    print(f"    Monkeys median:  {stats['mk_erp_median']:.6f}")
    print(
        f"    Minipigs active: {stats['mp_erp_active']}/{stats['mp_n_channels']} ({stats['mp_erp_active_pct']:.1f}%)"
    )
    print(
        f"    Monkeys active:  {stats['mk_erp_active']}/{stats['mk_n_channels']} ({stats['mk_erp_active_pct']:.1f}%)"
    )
    print("\n  Power Ratio SNR:")
    print(f"    Minipigs mean:   {stats['mp_power_mean']:.4f}")
    print(f"    Monkeys mean:    {stats['mk_power_mean']:.4f}")
    print(
        f"    Minipigs > 1.0:  {stats['mp_power_above1']}/{stats['mp_n_channels']} ({stats['mp_power_above1_pct']:.1f}%)"
    )
    print(
        f"    Monkeys > 1.0:   {stats['mk_power_above1']}/{stats['mk_n_channels']} ({stats['mk_power_above1_pct']:.1f}%)"
    )
    print("\n  Responsive Ratio:")
    print(f"    Minipigs mean:   {stats['mp_resp_mean'] * 100:.1f}%")
    print(f"    Monkeys mean:    {stats['mk_resp_mean'] * 100:.1f}%")
    print("\n  Statistical Tests:")
    print(f"    MW-U ERP:   p = {stats['mwu_erp_p']:.2e}")
    print(f"    MW-U Power: p = {stats['mwu_power_p']:.2e}")
    print(
        f"    KS Power:   D = {stats['ks_power_stat']:.4f}, p = {stats['ks_power_p']:.2e}"
    )


if __name__ == "__main__":
    main()
