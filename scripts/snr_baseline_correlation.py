"""Correlate SNR metrics with downstream decoding performance across baselines.

This script produces analysis for TWO separate report sections:

  Section 7 — SNR vs Decoding Performance:
    Does SNR (Evoked Potential or Power Ratio) correlate with decoding
    performance consistently across different baseline models?

  Section 8 — Session Consistency Across Baselines:
    Is session-level performance stable across baselines, or does it
    depend heavily on the model architecture?

Outputs (written to outputs/snr_analysis/):
  Section 7:
    - baseline_snr_correlation.png
    - baseline_snr_per_model_tuning.png
    - baseline_snr_per_model_maxsnr.png
    - baseline_correlation_table.csv
  Section 8:
    - session_consistency_heatmap.png
    - session_rank_stability.png
    - baseline_rank_correlation_heatmap.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASELINE_CSV = Path("outputs/baseline_results.csv")
SNR_SESSION_CSV = Path("outputs/snr_analysis/session_summary_table.csv")
CHANNEL_CSV = Path("outputs/snr_analysis/channel_quality_table.csv")
OUTPUT_DIR = Path("outputs/snr_analysis")

PERFORMANCE_METRIC = "f1"

SNR_METRICS = [
    ("max_broadband_snr", "Max Broadband SNR (ERP)"),
    ("max_tuning_metric", "Max Tonotopic Tuning"),
    ("mean_broadband_power_snr", "Mean Power Ratio SNR"),
    ("mean_lowfreq_power_snr", "Mean Low-Freq Power SNR"),
    ("mean_broadband_induced_snr", "Mean Induced Power SNR"),
    ("max_habituation_index", "Max Habituation Index"),
]

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_baseline_results(path: Path) -> pd.DataFrame:
    """Load baseline_results.csv and return a tidy DataFrame."""
    df = pd.read_csv(path)
    keep_cols = ["subject_id", "session_id", "baseline", PERFORMANCE_METRIC]
    extra = ["balanced_acc", "auroc", "loss"]
    keep_cols.extend([c for c in extra if c in df.columns])
    df = df[keep_cols].copy()
    df = df.dropna(subset=[PERFORMANCE_METRIC])
    return df


def load_snr_session_table(path: Path) -> pd.DataFrame:
    """Load the SNR session summary table."""
    return pd.read_csv(path)


def load_channel_table(path: Path) -> pd.DataFrame:
    """Load channel-level quality table for computing per-session max SNR."""
    return pd.read_csv(path)


def compute_session_max_snr(channel_df: pd.DataFrame) -> pd.DataFrame:
    """Compute max broadband SNR and max broadband power SNR per session."""
    agg = (
        channel_df
        .groupby("session_id")
        .agg(
            max_broadband_snr=("broadband_snr", "max"),
            mean_broadband_snr=("broadband_snr", "mean"),
            median_broadband_snr=("broadband_snr", "median"),
            max_broadband_power_snr=("broadband_power_snr", "max"),
        )
        .reset_index()
    )
    return agg


def merge_data(
    baseline_df: pd.DataFrame,
    snr_df: pd.DataFrame,
    channel_agg: pd.DataFrame,
) -> pd.DataFrame:
    """Merge baseline results with SNR session metrics and channel-level aggregates."""
    snr_full = snr_df.merge(channel_agg, on="session_id", how="left")
    merged = baseline_df.merge(snr_full, on="session_id", how="inner")
    return merged


# ---------------------------------------------------------------------------
# Section 7: Per-baseline correlation between SNR and F1
# ---------------------------------------------------------------------------


def compute_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman & Pearson correlations per baseline and SNR metric."""
    rows = []
    baselines = sorted(merged["baseline"].unique())

    for baseline in baselines:
        sub = merged[merged["baseline"] == baseline]
        for snr_col, snr_label in SNR_METRICS:
            if snr_col not in sub.columns:
                continue
            x = sub[snr_col].values
            y = sub[PERFORMANCE_METRIC].values
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            if len(x) < 5:
                continue

            sp_r, sp_p = stats.spearmanr(x, y)
            pe_r, pe_p = stats.pearsonr(x, y)
            rows.append({
                "baseline": baseline,
                "snr_metric": snr_label,
                "snr_col": snr_col,
                "n": len(x),
                "spearman_rho": sp_r,
                "spearman_p": sp_p,
                "pearson_r": pe_r,
                "pearson_p": pe_p,
            })

    for snr_col, snr_label in SNR_METRICS:
        if snr_col not in merged.columns:
            continue
        x = merged[snr_col].values
        y = merged[PERFORMANCE_METRIC].values
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < 5:
            continue
        sp_r, sp_p = stats.spearmanr(x, y)
        pe_r, pe_p = stats.pearsonr(x, y)
        rows.append({
            "baseline": "ALL (pooled)",
            "snr_metric": snr_label,
            "snr_col": snr_col,
            "n": len(x),
            "spearman_rho": sp_r,
            "spearman_p": sp_p,
            "pearson_r": pe_r,
            "pearson_p": pe_p,
        })

    return pd.DataFrame(rows)


def plot_baseline_snr_correlation(merged: pd.DataFrame, output_path: Path):
    """Scatter plots: F1 vs each SNR metric, colored by baseline."""
    baselines = sorted(merged["baseline"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(baselines)))
    color_map = dict(zip(baselines, colors))

    n_metrics = len(SNR_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 4.5))
    if n_metrics == 1:
        axes = [axes]

    for ax, (snr_col, snr_label) in zip(axes, SNR_METRICS):
        if snr_col not in merged.columns:
            continue
        for baseline in baselines:
            sub = merged[merged["baseline"] == baseline]
            ax.scatter(
                sub[snr_col],
                sub[PERFORMANCE_METRIC],
                c=[color_map[baseline]],
                label=baseline,
                alpha=0.7,
                s=30,
                edgecolors="none",
            )

        x_all = merged[snr_col].values
        y_all = merged[PERFORMANCE_METRIC].values
        valid = np.isfinite(x_all) & np.isfinite(y_all)
        x_fit, y_fit = x_all[valid], y_all[valid]
        if len(x_fit) > 2:
            sp_r, sp_p = stats.spearmanr(x_fit, y_fit)
            z = np.polyfit(x_fit, y_fit, 1)
            x_line = np.linspace(x_fit.min(), x_fit.max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.5, lw=1.5)
            ax.set_title(f"{snr_label}\n(ρ={sp_r:.3f}, p={sp_p:.2e})")

        ax.set_xlabel(snr_label)
        ax.set_ylabel(f"Macro {PERFORMANCE_METRIC.upper()}")

    axes[0].legend(loc="upper left", framealpha=0.8)
    fig.suptitle(
        "SNR vs Decoding Performance Across All Baselines",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_baseline_panels(
    merged: pd.DataFrame, snr_col: str, snr_label: str, output_path: Path
):
    """One panel per baseline showing F1 vs a specific SNR metric."""
    baselines = sorted(merged["baseline"].unique())

    n_base = len(baselines)
    ncols = 3
    nrows = int(np.ceil(n_base / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for i, baseline in enumerate(baselines):
        ax = axes[i]
        sub = merged[merged["baseline"] == baseline]
        x = sub[snr_col].values
        y = sub[PERFORMANCE_METRIC].values
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]

        ax.scatter(x, y, alpha=0.7, s=25, c="steelblue", edgecolors="none")
        if len(x) > 3:
            sp_r, sp_p = stats.spearmanr(x, y)
            z = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7, lw=1.2)
            sig_str = (
                "***"
                if sp_p < 0.001
                else ("**" if sp_p < 0.01 else ("*" if sp_p < 0.05 else "ns"))
            )
            ax.set_title(f"{baseline} (ρ={sp_r:.3f} {sig_str})")
        else:
            ax.set_title(baseline)
        ax.set_xlabel(snr_label)
        ax.set_ylabel("F1")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Per-Baseline: {snr_label} vs F1",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Section 8: Session consistency across baselines
# ---------------------------------------------------------------------------


def compute_session_consistency(merged: pd.DataFrame) -> dict:
    """Quantify how stable session rankings are across baselines."""
    pivot = merged.pivot_table(
        index="session_id",
        columns="baseline",
        values=PERFORMANCE_METRIC,
        aggfunc="mean",
    )
    pivot = pivot.dropna(thresh=3)

    baselines = pivot.columns.tolist()
    n_baselines = len(baselines)

    pairwise_rho = np.full((n_baselines, n_baselines), np.nan)
    for i in range(n_baselines):
        for j in range(n_baselines):
            valid = pivot[[baselines[i], baselines[j]]].dropna()
            if len(valid) >= 5:
                a = valid.iloc[:, 0].values.astype(float)
                b = valid.iloc[:, 1].values.astype(float)
                result = stats.spearmanr(a, b)
                rho_val = np.asarray(result.statistic).item()
                pairwise_rho[i, j] = rho_val

    mean_f1 = pivot.mean(axis=1)
    std_f1 = pivot.std(axis=1)
    cv = std_f1 / mean_f1.replace(0, np.nan)

    ranks = pivot.rank(ascending=False, axis=0)
    rank_std = ranks.std(axis=1)

    kendall_w = _kendall_w(pivot.values)

    return {
        "pivot": pivot,
        "pairwise_rho": pairwise_rho,
        "baselines": baselines,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "cv": cv,
        "rank_std": rank_std,
        "kendall_w": kendall_w,
    }


def _kendall_w(matrix: np.ndarray) -> float:
    """Kendall's W (coefficient of concordance) for rank agreement."""
    valid_mask = ~np.isnan(matrix)
    valid_rows = valid_mask.all(axis=1)
    matrix = matrix[valid_rows]
    if matrix.shape[0] < 3:
        return np.nan

    n_items = matrix.shape[0]
    k_judges = matrix.shape[1]

    ranks = np.zeros_like(matrix)
    for j in range(k_judges):
        ranks[:, j] = stats.rankdata(matrix[:, j])

    rank_sums = ranks.sum(axis=1)
    mean_rank_sum = rank_sums.mean()
    ss = np.sum((rank_sums - mean_rank_sum) ** 2)

    w = (12 * ss) / (k_judges**2 * (n_items**3 - n_items))
    return float(w)


def plot_session_heatmap(consistency: dict, output_path: Path):
    """Heatmap of F1 scores per session (rows) and baseline (columns)."""
    pivot = consistency["pivot"]
    sorted_idx = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot_sorted = pivot.loc[sorted_idx]

    short_names = [
        s.replace("_task-AcousStim_acq-", " ").replace("_desc-raw", "")
        for s in pivot_sorted.index
    ]

    fig, ax = plt.subplots(figsize=(8, max(6, len(short_names) * 0.3)))
    im = ax.imshow(
        pivot_sorted.values,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    ax.set_xticks(range(len(pivot_sorted.columns)))
    ax.set_xticklabels(pivot_sorted.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("Baseline Model")
    ax.set_ylabel("Session")
    ax.set_title(
        f"Session F1 Across Baselines (Kendall's W = {consistency['kendall_w']:.3f})",
        fontsize=11,
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="Macro F1", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_rank_stability(consistency: dict, output_path: Path):
    """Show how session ranks vary across baselines."""
    pivot = consistency["pivot"]
    sorted_idx = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot_sorted = pivot.loc[sorted_idx]

    short_names = [
        s.replace("_task-AcousStim_acq-", " ").replace("_desc-raw", "")
        for s in pivot_sorted.index
    ]

    baselines = pivot_sorted.columns.tolist()
    fig, ax = plt.subplots(figsize=(8, max(5, len(short_names) * 0.25)))

    for i, baseline in enumerate(baselines):
        col_vals = pivot_sorted[baseline].values
        ranks = stats.rankdata(-col_vals, nan_policy="omit")
        ax.scatter(
            ranks,
            range(len(short_names)),
            alpha=0.6,
            s=20,
            label=baseline,
        )

    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("Rank (1 = best)")
    ax.set_title(
        "Session Rank Across Baselines\n(tight clusters = consistent performance)",
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=7)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_pairwise_rank_correlation(consistency: dict, output_path: Path):
    """Heatmap of pairwise Spearman rank correlations between baselines."""
    rho_matrix = consistency["pairwise_rho"]
    baselines = consistency["baselines"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rho_matrix, cmap="coolwarm", vmin=0, vmax=1)
    ax.set_xticks(range(len(baselines)))
    ax.set_xticklabels(baselines, rotation=45, ha="right")
    ax.set_yticks(range(len(baselines)))
    ax.set_yticklabels(baselines)

    for i in range(len(baselines)):
        for j in range(len(baselines)):
            val = rho_matrix[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if val < 0.5 else "black",
                )

    ax.set_title(
        "Pairwise Spearman ρ Between Baselines\n(session rank agreement)",
        fontsize=10,
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("SNR vs Baseline Performance Correlation Analysis")
    print("=" * 60)

    baseline_df = load_baseline_results(BASELINE_CSV)
    snr_df = load_snr_session_table(SNR_SESSION_CSV)
    channel_df = load_channel_table(CHANNEL_CSV)
    channel_agg = compute_session_max_snr(channel_df)

    print(f"\nLoaded {len(baseline_df)} baseline result rows")
    print(f"  Baselines: {sorted(baseline_df['baseline'].unique())}")
    print(f"  Sessions in baseline: {baseline_df['session_id'].nunique()}")
    print(f"Loaded {len(snr_df)} SNR session rows")
    print(
        f"Loaded {len(channel_df)} channel rows -> {len(channel_agg)} session aggregates"
    )

    merged = merge_data(baseline_df, snr_df, channel_agg)
    print(
        f"Merged dataset: {len(merged)} rows ({merged['session_id'].nunique()} sessions)"
    )

    # Section 7: Correlations
    print("\n--- Section 7: SNR vs Performance Correlations ---")
    corr_df = compute_correlations(merged)
    corr_path = OUTPUT_DIR / "baseline_correlation_table.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"  Saved: {corr_path}")
    print(corr_df.to_string(index=False))

    print("\n--- Generating correlation plots ---")
    plot_baseline_snr_correlation(
        merged, OUTPUT_DIR / "baseline_snr_correlation.png"
    )
    plot_per_baseline_panels(
        merged,
        "max_tuning_metric",
        "Max Tonotopic Tuning",
        OUTPUT_DIR / "baseline_snr_per_model_tuning.png",
    )
    plot_per_baseline_panels(
        merged,
        "max_broadband_snr",
        "Max Broadband SNR (ERP)",
        OUTPUT_DIR / "baseline_snr_per_model_maxsnr.png",
    )

    # Section 8: Session consistency
    print("\n--- Section 8: Session Consistency Across Baselines ---")
    consistency = compute_session_consistency(merged)
    print(f"  Kendall's W = {consistency['kendall_w']:.4f}")
    triu_idx = np.triu_indices_from(consistency["pairwise_rho"], k=1)
    mean_rho = np.nanmean(consistency["pairwise_rho"][triu_idx])
    print(f"  Mean pairwise rank ρ = {mean_rho:.4f}")
    print(
        f"  Median CV of F1 across baselines = {consistency['cv'].median():.4f}"
    )

    plot_session_heatmap(
        consistency, OUTPUT_DIR / "session_consistency_heatmap.png"
    )
    plot_rank_stability(consistency, OUTPUT_DIR / "session_rank_stability.png")
    plot_pairwise_rank_correlation(
        consistency, OUTPUT_DIR / "baseline_rank_correlation_heatmap.png"
    )

    # Summary stats
    print("\n--- Summary for Report ---")
    print(f"  Total sessions: {merged['session_id'].nunique()}")
    print(f"  Total baselines: {merged['baseline'].nunique()}")
    print(f"  Kendall's W (concordance): {consistency['kendall_w']:.4f}")

    sig_corrs = corr_df[
        (corr_df["spearman_p"] < 0.05) & (corr_df["baseline"] != "ALL (pooled)")
    ]
    total_per_baseline = len(corr_df[corr_df["baseline"] != "ALL (pooled)"])
    print(
        f"  Significant per-baseline correlations: {len(sig_corrs)} / {total_per_baseline}"
    )

    return merged, corr_df, consistency


if __name__ == "__main__":
    main()
