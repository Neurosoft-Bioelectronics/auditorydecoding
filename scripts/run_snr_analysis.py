#!/usr/bin/env python
"""Run the full SNR signal-quality analysis on all minipig sessions.

This script implements the experimental protocol from SNR_EXP.md:
  1. Load every processed minipig session (.h5)
  2. Extract rest/stimulus epoch pairs
  3. Compute broadband and high-gamma SNR, responsive ratio, tuning metric
  4. Produce two summary tables (channel-level, session-level)
  5. Generate diagnostic plots (SNR histogram, top/bottom ERP comparison)

Usage::

    uv run python scripts/run_snr_analysis.py \\
        --data-dir /network/projects/neuro-galaxy/data/processed/neurosoft_minipigs_2026 \\
        --output-dir outputs/snr_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from auditorydecoding.analysis.snr import (
    load_session,
    extract_epochs,
    baseline_correct,
    apply_broadband_filter,
    apply_high_gamma_filter,
    apply_low_frequency_filter,
    build_channel_table,
    build_session_table,
    compute_channel_snr,
    compute_erp,
    compute_habituation_snr,
    compute_cumulative_erp_snr,
    compute_block_half_snr,
    compute_block_order_snr,
    identify_blocks,
)

DEFAULT_DATA_DIR = Path(
    "/network/projects/neuro-galaxy/data/processed/neurosoft_minipigs_2026"
)


# -----------------------------------------------------------------------
# Per-session analysis
# -----------------------------------------------------------------------


def analyse_session(
    h5_path: Path,
    snr_threshold: float = 0.5,
) -> tuple[dict, dict, np.ndarray, np.ndarray, float]:
    """Run the full analysis pipeline on a single session file.

    Returns
    -------
    channel_table : dict
        Per-channel metrics (one entry per ECOG channel).
    session_table : dict
        Aggregated session-level summary.
    bb_snr : np.ndarray
        Broadband SNR per channel (for plotting).
    erp : np.ndarray
        Broadband ERP per channel, shape (n_channels, n_time).
    sampling_rate : float
    bb_epochs : EpochArrays
        Broadband-filtered epochs (for habituation analysis).
    """
    print(f"  Loading {h5_path.name} \u2026")
    session = load_session(h5_path)

    epochs = extract_epochs(session)
    epochs = baseline_correct(epochs)

    bb_epochs = apply_broadband_filter(epochs, session.sampling_rate)
    hg_epochs = apply_high_gamma_filter(epochs, session.sampling_rate)
    lf_epochs = apply_low_frequency_filter(epochs, session.sampling_rate)

    ch_table = build_channel_table(
        session,
        bb_epochs,
        hg_epochs,
        snr_threshold=snr_threshold,
        low_freq_epochs=lf_epochs,
    )
    ses_table = build_session_table(ch_table)

    bb_snr = compute_channel_snr(bb_epochs)
    erp = compute_erp(bb_epochs)

    return ch_table, ses_table, bb_snr, erp, session.sampling_rate, bb_epochs


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------


def plot_snr_histogram(
    channel_df: pd.DataFrame,
    output_path: Path,
    snr_threshold: float = 0.5,
) -> None:
    """Histogram of broadband SNR across all channels and sessions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    snr_values = channel_df["broadband_snr"].values

    ax.hist(snr_values, bins=50, edgecolor="black", alpha=0.7, color="#4c72b0")
    ax.axvline(
        snr_threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold = {snr_threshold}",
    )
    ax.set_xlabel("Broadband SNR")
    ax.set_ylabel("Number of channels")
    ax.set_title("Distribution of channel-level broadband SNR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved SNR histogram → {output_path}")


def plot_top_vs_bottom_erp(
    channel_df: pd.DataFrame,
    erp_dict: dict[str, np.ndarray],
    sfreq_dict: dict[str, float],
    output_path: Path,
    n: int = 5,
) -> None:
    """Grand-average ERP for top-N vs bottom-N SNR channels (across sessions).

    For each of the top/bottom channels we pick the first session in which it
    appears so that we have a concrete ERP waveform to show.
    """
    sorted_df = channel_df.sort_values("broadband_snr", ascending=False)
    top_rows = sorted_df.head(n)
    bottom_rows = sorted_df.tail(n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, rows, title in [
        (axes[0], top_rows, f"Top {n} channels (highest SNR)"),
        (axes[1], bottom_rows, f"Bottom {n} channels (lowest SNR)"),
    ]:
        for _, row in rows.iterrows():
            sid = row["session_id"]
            ch_idx = row["_ch_idx"]
            erp = erp_dict[sid]
            sfreq = sfreq_dict[sid]
            t_ms = np.arange(erp.shape[-1]) / sfreq * 1000
            label = f"{sid[-30:]}\n{row['channel_id']} (SNR={row['broadband_snr']:.2f})"
            ax.plot(t_ms, erp[ch_idx], label=label, alpha=0.8)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title(title)
        ax.legend(fontsize=6, loc="upper right")

    fig.suptitle("Grand-average ERP: top vs bottom SNR channels", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved ERP comparison → {output_path}")


def plot_lowfreq_vs_broadband_snr(
    channel_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Scatter plot of low-frequency SNR vs broadband SNR per channel."""
    fig, ax = plt.subplots(figsize=(8, 7))

    bb = channel_df["broadband_snr"].values
    lf = channel_df["lowfreq_snr"].values

    ax.scatter(bb, lf, alpha=0.4, s=18, color="#4c72b0", edgecolors="none")

    lim = max(bb.max(), lf.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, linewidth=1, label="y = x")

    ax.set_xlabel("Broadband SNR (1\u2013300 Hz)")
    ax.set_ylabel("Low-Frequency SNR (1\u201370 Hz)")
    ax.set_title("Low-Frequency vs. Broadband SNR (per channel)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved LF vs BB scatter → {output_path}")


def plot_lowfreq_snr_histogram(
    channel_df: pd.DataFrame,
    output_path: Path,
    snr_threshold: float = 0.5,
) -> None:
    """Overlaid histograms comparing broadband and low-frequency SNR."""
    fig, ax = plt.subplots(figsize=(9, 5))

    bb = channel_df["broadband_snr"].values
    lf = channel_df["lowfreq_snr"].values

    max_val = max(bb.max(), lf.max())
    bins = np.linspace(0, min(max_val, 5.0), 60)

    ax.hist(
        bb,
        bins=bins,
        alpha=0.5,
        color="#4c72b0",
        edgecolor="black",
        linewidth=0.5,
        label="Broadband (1\u2013300 Hz)",
    )
    ax.hist(
        lf,
        bins=bins,
        alpha=0.5,
        color="#e74c3c",
        edgecolor="black",
        linewidth=0.5,
        label="Low-Freq (1\u201370 Hz)",
    )
    ax.axvline(
        snr_threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold = {snr_threshold}",
    )

    ax.set_xlabel("SNR")
    ax.set_ylabel("Number of channels")
    ax.set_title("SNR Distribution: Broadband vs. Low-Frequency Filtering")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved LF vs BB histogram → {output_path}")


def plot_lowfreq_session_comparison(
    session_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Bar chart comparing active channel counts across filter bands per session."""
    if "lowfreq_active_channels" not in session_df.columns:
        return

    has_any = (session_df["active_channels"] > 0) | (
        session_df["lowfreq_active_channels"] > 0
    )
    df = session_df[has_any].copy()
    if df.empty:
        return

    df["short_id"] = df["session_id"].apply(
        lambda s: s[:35] + "…" if len(s) > 35 else s
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(
        x - w / 2,
        df["active_channels"],
        w,
        label="Broadband (1\u2013300 Hz)",
        color="#4c72b0",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + w / 2,
        df["lowfreq_active_channels"],
        w,
        label="Low-Freq (1\u201370 Hz)",
        color="#e74c3c",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["short_id"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Active Channels")
    ax.set_title(
        "Active Channel Count by Filter Band (sessions with any active channels)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved session comparison → {output_path}")


# -----------------------------------------------------------------------
# Habituation plots (RQ7)
# -----------------------------------------------------------------------


def plot_habituation_quartiles(
    quartile_snrs: np.ndarray,
    output_path: Path,
    n_splits: int = 4,
) -> None:
    """Bar chart of mean evoked SNR per chronological quartile."""
    mean_per_q = quartile_snrs.mean(axis=1)  # (n_splits,)
    sem_per_q = quartile_snrs.std(axis=1) / np.sqrt(quartile_snrs.shape[1])

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"Q{i + 1}" for i in range(n_splits)]
    x = np.arange(n_splits)
    ax.bar(
        x,
        mean_per_q,
        yerr=sem_per_q,
        color=plt.cm.Blues(np.linspace(0.4, 0.9, n_splits)),
        edgecolor="black",
        linewidth=0.5,
        capsize=4,
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Trial Quartile (Q1 = earliest)")
    ax.set_ylabel("Mean Evoked SNR")
    ax.set_title("Evoked SNR by Chronological Trial Quartile")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved habituation quartiles \u2192 {output_path}")


def plot_cumulative_snr_curves(
    cumulative_snrs: dict[str, np.ndarray],
    sfreq_dict: dict[str, float],
    output_path: Path,
    top_n: int = 10,
    step: int = 5,
) -> None:
    """Line plot: ERP SNR vs number of trials for top-N channels across sessions."""
    all_curves = []
    for sid, curves in cumulative_snrs.items():
        max_snr_per_ch = curves.max(axis=0)
        top_ch = np.argsort(max_snr_per_ch)[-top_n:]
        for ch_idx in top_ch:
            all_curves.append(curves[:, ch_idx])

    if not all_curves:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for curve in all_curves:
        n_steps = len(curve)
        trial_counts = [(i + 1) * step for i in range(n_steps - 1)]
        trial_counts.append(trial_counts[-1] + step if n_steps > 1 else step)
        ax.plot(
            trial_counts[:n_steps],
            curve,
            alpha=0.3,
            color="steelblue",
            linewidth=0.8,
        )

    if all_curves:
        max_len = max(len(c) for c in all_curves)
        padded = np.full((len(all_curves), max_len), np.nan)
        for i, c in enumerate(all_curves):
            padded[i, : len(c)] = c
        mean_curve = np.nanmean(padded, axis=0)
        trial_counts = [(i + 1) * step for i in range(max_len)]
        ax.plot(
            trial_counts,
            mean_curve,
            color="red",
            linewidth=2,
            label="Mean across top channels",
        )

    ax.set_xlabel("Number of Trials Included")
    ax.set_ylabel("Cumulative ERP SNR")
    ax.set_title("Cumulative ERP SNR vs. Trial Count (top channels)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved cumulative SNR curves \u2192 {output_path}")


def plot_habituation_index_distribution(
    channel_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Histogram of per-channel habituation index."""
    hi = channel_df["habituation_index"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(hi, bins=50, edgecolor="black", alpha=0.7, color="#e67e22")
    ax.axvline(
        0, color="black", linestyle="--", linewidth=1.5, label="HI = 0 (stable)"
    )
    ax.axvline(
        0.1,
        color="red",
        linestyle=":",
        linewidth=1.2,
        label="HI = 0.1 (mild habituation)",
    )
    ax.axvline(
        -0.1,
        color="blue",
        linestyle=":",
        linewidth=1.2,
        label="HI = -0.1 (mild sensitization)",
    )

    ax.set_xlabel("Habituation Index")
    ax.set_ylabel("Number of Channels")
    ax.set_title("Distribution of Per-Channel Habituation Index")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved habituation index histogram \u2192 {output_path}")


# -----------------------------------------------------------------------
# Block-level habituation plots (RQ8)
# -----------------------------------------------------------------------


def plot_within_block_habituation(
    all_block_halves: list[dict],
    output_path: Path,
) -> None:
    """Paired comparison: first-half vs second-half SNR within blocks."""
    first_means = []
    second_means = []
    for bh in all_block_halves:
        mask = ~np.isnan(bh["first_half_snr"]).any(axis=1)
        if mask.any():
            first_means.extend(
                np.nanmean(bh["first_half_snr"][mask], axis=1).tolist()
            )
            second_means.extend(
                np.nanmean(bh["second_half_snr"][mask], axis=1).tolist()
            )

    if not first_means:
        return

    first_mean_per_block = np.array(first_means)
    second_mean_per_block = np.array(second_means)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: paired scatter
    ax = axes[0]
    lim = max(first_mean_per_block.max(), second_mean_per_block.max()) * 1.1
    ax.scatter(
        first_mean_per_block,
        second_mean_per_block,
        alpha=0.3,
        s=20,
        color="#e67e22",
        edgecolors="none",
    )
    ax.plot([0, lim], [0, lim], "k--", alpha=0.4, linewidth=1, label="y = x")
    ax.set_xlabel("First Half of Block — Mean SNR")
    ax.set_ylabel("Second Half of Block — Mean SNR")
    ax.set_title("Within-Block Habituation (per block)")
    ax.legend()

    # Panel 2: bar chart of grand means
    ax2 = axes[1]
    means = [first_mean_per_block.mean(), second_mean_per_block.mean()]
    sems = [
        first_mean_per_block.std() / np.sqrt(len(first_mean_per_block)),
        second_mean_per_block.std() / np.sqrt(len(second_mean_per_block)),
    ]
    bars = ax2.bar(
        [0, 1],
        means,
        yerr=sems,
        color=["#3498db", "#e74c3c"],
        edgecolor="black",
        linewidth=0.5,
        capsize=6,
        alpha=0.85,
    )
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["First Half", "Second Half"])
    ax2.set_ylabel("Mean Evoked SNR")
    ax2.set_title("Grand Mean: First vs Second Half of Each Block")
    for bar, m in zip(bars, means):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + sems[0] * 0.3,
            f"{m:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.suptitle("Within-Block Habituation Analysis", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved within-block habituation → {output_path}")


def plot_across_block_habituation(
    all_block_orders: list[dict],
    output_path: Path,
) -> None:
    """SNR as a function of block index (chronological order)."""
    max_blocks = max(len(bo["per_block_snr"]) for bo in all_block_orders)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: individual session curves (mean across channels per block)
    ax = axes[0]
    all_block_means = []
    for bo in all_block_orders:
        block_means = np.nanmean(bo["per_block_snr"], axis=1)
        n_blocks = len(block_means)
        ax.plot(
            range(1, n_blocks + 1),
            block_means,
            alpha=0.15,
            color="steelblue",
            linewidth=0.8,
        )
        all_block_means.append(block_means)

    # Grand mean curve (padded)
    padded = np.full((len(all_block_means), max_blocks), np.nan)
    for i, bm in enumerate(all_block_means):
        padded[i, : len(bm)] = bm
    grand_mean = np.nanmean(padded, axis=0)
    grand_sem = np.nanstd(padded, axis=0) / np.sqrt(
        np.sum(~np.isnan(padded), axis=0)
    )
    x = np.arange(1, max_blocks + 1)
    valid = ~np.isnan(grand_mean)
    ax.plot(
        x[valid],
        grand_mean[valid],
        color="red",
        linewidth=2,
        label="Grand mean",
    )
    ax.fill_between(
        x[valid],
        (grand_mean - grand_sem)[valid],
        (grand_mean + grand_sem)[valid],
        color="red",
        alpha=0.2,
    )
    ax.set_xlabel("Block Index (chronological)")
    ax.set_ylabel("Mean Evoked SNR (across channels)")
    ax.set_title("SNR by Block Order (all sessions)")
    ax.legend()

    # Panel 2: first-third vs middle-third vs last-third bars
    ax2 = axes[1]
    thirds_snr = [[], [], []]
    for bo in all_block_orders:
        snr = bo["per_block_snr"]  # (n_blocks, n_channels)
        n = len(snr)
        if n < 3:
            continue
        t1 = n // 3
        t2 = 2 * n // 3
        thirds_snr[0].append(np.nanmean(snr[:t1]))
        thirds_snr[1].append(np.nanmean(snr[t1:t2]))
        thirds_snr[2].append(np.nanmean(snr[t2:]))

    if all(len(t) > 0 for t in thirds_snr):
        means = [np.mean(t) for t in thirds_snr]
        sems = [np.std(t) / np.sqrt(len(t)) for t in thirds_snr]
        colors = ["#3498db", "#f39c12", "#e74c3c"]
        labels = ["Early\nBlocks", "Middle\nBlocks", "Late\nBlocks"]
        bars = ax2.bar(
            range(3),
            means,
            yerr=sems,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            capsize=6,
            alpha=0.85,
        )
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("Mean Evoked SNR")
        ax2.set_title("SNR by Block Position (thirds)")
        for bar, m in zip(bars, means):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{m:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    fig.suptitle("Across-Block Habituation Analysis", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved across-block habituation → {output_path}")


def plot_block_erp_comparison(
    bb_epochs_dict: dict,
    sfreq_dict: dict,
    output_path: Path,
    n_sessions: int = 4,
) -> None:
    """ERP waveforms from early vs late within-block trials for top sessions."""

    session_snrs = {}
    for sid, ep in bb_epochs_dict.items():
        session_snrs[sid] = np.mean(compute_channel_snr(ep))

    top_sids = sorted(session_snrs, key=session_snrs.get, reverse=True)[
        :n_sessions
    ]

    if not top_sids:
        return

    fig, axes = plt.subplots(
        len(top_sids), 1, figsize=(12, 3.5 * len(top_sids)), squeeze=False
    )

    for row, sid in enumerate(top_sids):
        ax = axes[row, 0]
        ep = bb_epochs_dict[sid]
        sfreq = sfreq_dict[sid]
        blocks = identify_blocks(ep.stim_labels)

        snr_all = compute_channel_snr(ep)
        best_ch = np.argmax(snr_all)

        # Aggregate ERPs from first-half and second-half of all blocks
        first_half_trials = []
        second_half_trials = []
        for start, end, _ in blocks:
            n = end - start
            if n < 4:
                continue
            mid = start + n // 2
            first_half_trials.append(ep.stimulus[start:mid, best_ch, :])
            second_half_trials.append(ep.stimulus[mid:end, best_ch, :])

        if not first_half_trials:
            continue

        first_erp = np.concatenate(first_half_trials, axis=0).mean(axis=0)
        second_erp = np.concatenate(second_half_trials, axis=0).mean(axis=0)

        t_ms = np.arange(len(first_erp)) / sfreq * 1000
        ax.plot(
            t_ms,
            first_erp,
            color="#3498db",
            linewidth=1.5,
            label="First half of blocks",
        )
        ax.plot(
            t_ms,
            second_erp,
            color="#e74c3c",
            linewidth=1.5,
            label="Second half of blocks",
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        short_sid = sid[:40] + "…" if len(sid) > 40 else sid
        ax.set_title(
            f"{short_sid} — Best channel (ch {best_ch}, SNR={snr_all[best_ch]:.3f})"
        )
        ax.legend(fontsize=8)

    fig.suptitle(
        "ERP: First vs Second Half of Blocks (best channel per session)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved block ERP comparison → {output_path}")


# -----------------------------------------------------------------------
# Power Ratio SNR plots (RQ6)
# -----------------------------------------------------------------------


def plot_power_ratio_histogram(
    channel_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Overlaid histograms of evoked-SNR vs power-ratio SNR (broadband)."""
    fig, ax = plt.subplots(figsize=(9, 5))

    evoked = channel_df["broadband_snr"].values
    power = channel_df["broadband_power_snr"].values

    max_val = max(np.percentile(evoked, 99), np.percentile(power, 99))
    bins = np.linspace(0, min(max_val, 5.0), 60)

    ax.hist(
        evoked,
        bins=bins,
        alpha=0.5,
        color="#4c72b0",
        edgecolor="black",
        linewidth=0.5,
        label="Evoked SNR (phase-locked)",
    )
    ax.hist(
        power,
        bins=bins,
        alpha=0.5,
        color="#2ecc71",
        edgecolor="black",
        linewidth=0.5,
        label="Power Ratio SNR (phase-insensitive)",
    )
    ax.axvline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Power Ratio = 1.0 (stimulus = rest)",
    )
    ax.axvline(
        0.5,
        color="red",
        linestyle=":",
        linewidth=1.2,
        label="Evoked SNR threshold = 0.5",
    )

    ax.set_xlabel("SNR")
    ax.set_ylabel("Number of channels")
    ax.set_title("Evoked SNR vs. Power Ratio SNR Distribution (Broadband)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved power ratio histogram \u2192 {output_path}")


def plot_power_ratio_scatter(
    channel_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Scatter: evoked SNR (x) vs power ratio SNR (y), colored by status."""
    fig, ax = plt.subplots(figsize=(8, 7))

    active = channel_df["status"] == "Active"
    evoked = channel_df["broadband_snr"].values
    power = channel_df["broadband_power_snr"].values

    ax.scatter(
        evoked[~active],
        power[~active],
        alpha=0.3,
        s=18,
        color="#aaaaaa",
        edgecolors="none",
        label="Dead channels",
    )
    ax.scatter(
        evoked[active],
        power[active],
        alpha=0.8,
        s=40,
        color="#e74c3c",
        edgecolors="black",
        linewidth=0.5,
        label="Active channels",
        zorder=5,
    )

    ax.axhline(
        1.0,
        color="#2ecc71",
        linestyle="--",
        alpha=0.7,
        linewidth=1,
        label="Power Ratio = 1",
    )
    ax.axvline(
        0.5,
        color="#4c72b0",
        linestyle=":",
        alpha=0.7,
        linewidth=1,
        label="Evoked SNR = 0.5",
    )

    ax.set_xlabel("Evoked SNR (phase-locked)")
    ax.set_ylabel("Power Ratio SNR (phase-insensitive)")
    ax.set_title("Evoked SNR vs. Power Ratio SNR (per channel)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved power ratio scatter \u2192 {output_path}")


def plot_power_ratio_by_band(
    channel_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Grouped bar chart: mean power ratio SNR across filter bands."""
    bands = ["broadband_power_snr", "high_gamma_power_snr"]
    labels = ["Broadband\n(1\u2013300 Hz)", "High-Gamma\n(70\u2013150 Hz)"]
    colors = ["#4c72b0", "#9b59b6"]

    if "lowfreq_power_snr" in channel_df.columns:
        bands.append("lowfreq_power_snr")
        labels.append("Low-Freq\n(1\u201370 Hz)")
        colors.append("#e74c3c")

    active = channel_df["status"] == "Active"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, mask, title in [
        (axes[0], np.ones(len(channel_df), dtype=bool), "All Channels"),
        (axes[1], active.values, "Active Channels Only"),
    ]:
        if mask.sum() == 0:
            ax.set_title(f"{title}\n(no channels)")
            continue

        means = [channel_df.loc[mask, b].mean() for b in bands]
        sems = [channel_df.loc[mask, b].sem() for b in bands]

        x = np.arange(len(bands))
        bars = ax.bar(
            x,
            means,
            yerr=sems,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            capsize=4,
            alpha=0.85,
        )
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Mean Power Ratio SNR")
        ax.set_title(title)

        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{m:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    fig.suptitle("Power Ratio SNR by Frequency Band", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved power ratio by band \u2192 {output_path}")


def plot_power_ratio_session_bars(
    session_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Per-session mean power ratio SNR bar chart."""
    fig, ax = plt.subplots(figsize=(14, 5))

    df = session_df.sort_values(
        "mean_broadband_power_snr", ascending=False
    ).copy()
    df["short_id"] = df["session_id"].apply(
        lambda s: s[:35] + "\u2026" if len(s) > 35 else s
    )

    x = np.arange(len(df))
    bars = ax.bar(
        x,
        df["mean_broadband_power_snr"],
        color="#2ecc71",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
    )
    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        alpha=0.6,
        linewidth=1.2,
        label="Power Ratio = 1.0",
    )

    for i, (_, row) in enumerate(df.iterrows()):
        if row["active_channels"] > 0:
            bars[i].set_color("#e74c3c")
            bars[i].set_alpha(1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(df["short_id"], rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("Mean Power Ratio SNR (broadband)")
    ax.set_title(
        "Per-Session Mean Power Ratio SNR (red = sessions with active channels)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved power ratio session bars \u2192 {output_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing processed .h5 session files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/snr_analysis"),
        help="Directory for CSV tables and plots.",
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=0.5,
        help="SNR threshold to classify channels as Active/Dead.",
    )
    parser.add_argument(
        "--desc-filter",
        type=str,
        default="desc-raw",
        help="Only process sessions whose filename contains this substring.",
    )
    args = parser.parse_args()

    h5_files = sorted(args.data_dir.glob("*.h5"))
    if args.desc_filter:
        h5_files = [f for f in h5_files if args.desc_filter in f.name]

    if not h5_files:
        print(f"No .h5 files found in {args.data_dir}")
        return

    print(f"Found {len(h5_files)} session files to analyse.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_channel_rows: list[dict] = []
    all_session_rows: list[dict] = []
    erp_dict: dict[str, np.ndarray] = {}
    sfreq_dict: dict[str, float] = {}
    bb_epochs_dict: dict[str, object] = {}

    for h5_path in h5_files:
        try:
            ch_table, ses_table, bb_snr, erp, sfreq, bb_epochs = (
                analyse_session(
                    h5_path,
                    snr_threshold=args.snr_threshold,
                )
            )
        except Exception as exc:
            print(f"  \u26a0 Skipping {h5_path.name}: {exc}")
            continue

        sid = ch_table["session_id"][0]
        n_ch = len(ch_table["channel_id"])
        for j in range(n_ch):
            row = {
                k: v[j] if hasattr(v, "__getitem__") else v
                for k, v in ch_table.items()
            }
            row["_ch_idx"] = j
            all_channel_rows.append(row)

        all_session_rows.append(ses_table)
        erp_dict[sid] = erp
        sfreq_dict[sid] = sfreq
        bb_epochs_dict[sid] = bb_epochs

    if not all_channel_rows:
        print("No sessions were processed successfully.")
        return

    channel_df = pd.DataFrame(all_channel_rows)
    session_df = pd.DataFrame(all_session_rows)

    # Save tables
    ch_csv = args.output_dir / "channel_quality_table.csv"
    ses_csv = args.output_dir / "session_summary_table.csv"
    channel_df.drop(columns=["_ch_idx"]).to_csv(ch_csv, index=False)
    session_df.to_csv(ses_csv, index=False)
    print(f"\nSaved channel table  → {ch_csv}  ({len(channel_df)} rows)")
    print(f"Saved session table  → {ses_csv}  ({len(session_df)} rows)")

    # Plots
    plot_snr_histogram(
        channel_df,
        args.output_dir / "snr_histogram.png",
        snr_threshold=args.snr_threshold,
    )
    plot_top_vs_bottom_erp(
        channel_df,
        erp_dict,
        sfreq_dict,
        args.output_dir / "erp_top_vs_bottom.png",
    )

    if "lowfreq_snr" in channel_df.columns:
        plot_lowfreq_vs_broadband_snr(
            channel_df,
            args.output_dir / "lowfreq_vs_broadband_scatter.png",
        )
        plot_lowfreq_snr_histogram(
            channel_df,
            args.output_dir / "lowfreq_vs_broadband_histogram.png",
            snr_threshold=args.snr_threshold,
        )
        plot_lowfreq_session_comparison(
            session_df,
            args.output_dir / "lowfreq_session_comparison.png",
        )

        # Print low-freq summary statistics
        lf_active = (channel_df["lowfreq_status"] == "Active").sum()
        bb_active = (channel_df["status"] == "Active").sum()
        print("\n--- Low-Frequency Filter Analysis ---")
        print(f"  Broadband active channels: {bb_active}/{len(channel_df)}")
        print(f"  Low-freq active channels:  {lf_active}/{len(channel_df)}")
        print(
            f"  Median broadband SNR: {channel_df['broadband_snr'].median():.6f}"
        )
        print(
            f"  Median low-freq SNR:  {channel_df['lowfreq_snr'].median():.6f}"
        )
        print(
            f"  Mean broadband SNR:   {channel_df['broadband_snr'].mean():.6f}"
        )
        print(f"  Mean low-freq SNR:    {channel_df['lowfreq_snr'].mean():.6f}")

        from scipy.stats import spearmanr, wilcoxon

        rho, pval = spearmanr(
            channel_df["broadband_snr"], channel_df["lowfreq_snr"]
        )
        print(f"  Spearman(BB, LF):     rho={rho:.4f}, p={pval:.2e}")

        try:
            stat, wp = wilcoxon(
                channel_df["lowfreq_snr"], channel_df["broadband_snr"]
            )
            print(f"  Wilcoxon signed-rank: stat={stat:.1f}, p={wp:.2e}")
        except ValueError:
            pass

        lf_resp = channel_df["lowfreq_resp_ratio"]
        bb_resp = channel_df["broadband_resp_ratio"]
        print(f"  Mean broadband resp ratio: {bb_resp.mean():.4f}")
        print(f"  Mean low-freq resp ratio:  {lf_resp.mean():.4f}")

    # ------------------------------------------------------------------
    # RQ6: Power Ratio SNR (phase-insensitive)
    # ------------------------------------------------------------------
    if "broadband_power_snr" in channel_df.columns:
        from scipy.stats import spearmanr as _spearmanr, wilcoxon as _wilcoxon

        print("\n--- RQ6: Power Ratio SNR (Phase-Insensitive) ---")

        bb_power = channel_df["broadband_power_snr"]
        bb_evoked = channel_df["broadband_snr"]
        active_mask = channel_df["status"] == "Active"

        print(f"  Mean evoked SNR (all):       {bb_evoked.mean():.6f}")
        print(f"  Mean power ratio SNR (all):  {bb_power.mean():.6f}")
        print(f"  Median evoked SNR:           {bb_evoked.median():.6f}")
        print(f"  Median power ratio SNR:      {bb_power.median():.6f}")

        above_1 = (bb_power > 1.0).sum()
        print(
            f"  Channels with power ratio > 1.0: {above_1}/{len(bb_power)} "
            f"({100 * above_1 / len(bb_power):.1f}%)"
        )

        pct_higher = (bb_power > bb_evoked).sum()
        print(
            f"  Channels where power ratio > evoked: {pct_higher}/{len(bb_power)} "
            f"({100 * pct_higher / len(bb_power):.1f}%)"
        )

        if active_mask.any():
            print("\n  Active channels:")
            print(
                f"    Mean evoked SNR:      {bb_evoked[active_mask].mean():.4f}"
            )
            print(
                f"    Mean power ratio SNR: {bb_power[active_mask].mean():.4f}"
            )

        rho, pval = _spearmanr(bb_evoked, bb_power)
        print(f"  Spearman(evoked, power): rho={rho:.4f}, p={pval:.2e}")

        try:
            stat, wp = _wilcoxon(bb_power, bb_evoked)
            print(
                f"  Wilcoxon signed-rank (power vs evoked): stat={stat:.1f}, p={wp:.2e}"
            )
        except ValueError:
            pass

        if "high_gamma_power_snr" in channel_df.columns:
            hg_power = channel_df["high_gamma_power_snr"]
            hg_above = (hg_power > 1.0).sum()
            print("\n  High-gamma power ratio:")
            print(f"    Mean:  {hg_power.mean():.6f}")
            print(
                f"    > 1.0: {hg_above}/{len(hg_power)} ({100 * hg_above / len(hg_power):.1f}%)"
            )

        if "lowfreq_power_snr" in channel_df.columns:
            lf_power = channel_df["lowfreq_power_snr"]
            lf_above = (lf_power > 1.0).sum()
            print("\n  Low-freq power ratio:")
            print(f"    Mean:  {lf_power.mean():.6f}")
            print(
                f"    > 1.0: {lf_above}/{len(lf_power)} ({100 * lf_above / len(lf_power):.1f}%)"
            )

        # Plots
        plot_power_ratio_histogram(
            channel_df,
            args.output_dir / "power_ratio_histogram.png",
        )
        plot_power_ratio_scatter(
            channel_df,
            args.output_dir / "power_ratio_scatter.png",
        )
        plot_power_ratio_by_band(
            channel_df,
            args.output_dir / "power_ratio_by_band.png",
        )
        plot_power_ratio_session_bars(
            session_df,
            args.output_dir / "power_ratio_session_bars.png",
        )

    # ------------------------------------------------------------------
    # Induced Power SNR (Hypothesis A)
    # ------------------------------------------------------------------
    if "broadband_induced_snr" in channel_df.columns:
        from scipy.stats import spearmanr as _sp2

        print("\n--- Hypothesis A: Induced Power SNR ---")
        bb_induced = channel_df["broadband_induced_snr"]
        bb_evoked = channel_df["broadband_snr"]
        bb_power = channel_df["broadband_power_snr"]

        print(f"  Mean induced SNR (all):    {bb_induced.mean():.6f}")
        print(f"  Median induced SNR (all):  {bb_induced.median():.6f}")
        print(f"  Mean evoked SNR (all):     {bb_evoked.mean():.6f}")
        print(f"  Mean power ratio SNR (all):{bb_power.mean():.6f}")

        induced_dom = (bb_induced > bb_evoked).sum()
        print(
            f"  Channels where induced > evoked: {induced_dom}/{len(bb_induced)} "
            f"({100 * induced_dom / len(bb_induced):.1f}%)"
        )

        rho_ip, p_ip = _sp2(bb_induced, bb_power)
        print(
            f"  Spearman(induced, power ratio): rho={rho_ip:.4f}, p={p_ip:.2e}"
        )
        rho_ie, p_ie = _sp2(bb_induced, bb_evoked)
        print(
            f"  Spearman(induced, evoked):       rho={rho_ie:.4f}, p={p_ie:.2e}"
        )

    # ------------------------------------------------------------------
    # RQ7: Habituation Analysis (Hypothesis B)
    # ------------------------------------------------------------------
    if "habituation_index" in channel_df.columns and bb_epochs_dict:
        from scipy.stats import wilcoxon as _wil2, spearmanr as _sp3

        print("\n--- RQ7: Habituation Analysis (Hypothesis B) ---")

        hi = channel_df["habituation_index"]
        print(f"  Mean HI:   {hi.mean():.6f}")
        print(f"  Median HI: {hi.median():.6f}")
        hi_pos = (hi > 0.1).sum()
        hi_neg = (hi < -0.1).sum()
        print(
            f"  Channels with HI > 0.1 (habituation): {hi_pos}/{len(hi)} "
            f"({100 * hi_pos / len(hi):.1f}%)"
        )
        print(
            f"  Channels with HI < -0.1 (sensitization): {hi_neg}/{len(hi)} "
            f"({100 * hi_neg / len(hi):.1f}%)"
        )

        all_quartile_snrs = []
        cumulative_snrs_dict: dict[str, np.ndarray] = {}
        for sid, bb_ep in bb_epochs_dict.items():
            q_snr = compute_habituation_snr(bb_ep, n_splits=4)
            all_quartile_snrs.append(q_snr)
            cum_snr = compute_cumulative_erp_snr(bb_ep, step=5)
            cumulative_snrs_dict[sid] = cum_snr

        if all_quartile_snrs:
            stacked_q = np.concatenate(
                all_quartile_snrs, axis=1
            )  # (4, total_channels)
            q1_snr = stacked_q[0]
            q4_snr = stacked_q[-1]
            print(f"\n  Mean Q1 SNR: {q1_snr.mean():.6f}")
            print(f"  Mean Q4 SNR: {q4_snr.mean():.6f}")

            try:
                stat, wp = _wil2(q1_snr, q4_snr)
                print(f"  Wilcoxon Q1 vs Q4: stat={stat:.1f}, p={wp:.2e}")
            except ValueError as e:
                print(f"  Wilcoxon Q1 vs Q4: could not compute ({e})")

            rho_hi_ev, p_hi_ev = _sp3(hi, channel_df["broadband_snr"])
            print(
                f"  Spearman(HI, evoked SNR): rho={rho_hi_ev:.4f}, p={p_hi_ev:.2e}"
            )

            plot_habituation_quartiles(
                stacked_q,
                args.output_dir / "habituation_quartiles.png",
            )

        plot_habituation_index_distribution(
            channel_df,
            args.output_dir / "habituation_index_histogram.png",
        )

        if cumulative_snrs_dict:
            plot_cumulative_snr_curves(
                cumulative_snrs_dict,
                sfreq_dict,
                args.output_dir / "cumulative_snr_curves.png",
            )

    # ------------------------------------------------------------------
    # RQ8: Block-Level Habituation Analysis
    # ------------------------------------------------------------------
    if bb_epochs_dict:
        from scipy.stats import wilcoxon as _wil3, spearmanr as _sp4

        print("\n--- RQ8: Block-Level Habituation Analysis ---")

        all_block_halves = []
        all_block_orders = []
        total_blocks = 0

        for sid, bb_ep in bb_epochs_dict.items():
            bh = compute_block_half_snr(bb_ep)
            bo = compute_block_order_snr(bb_ep)
            all_block_halves.append(bh)
            all_block_orders.append(bo)
            total_blocks += len(bh["blocks"])

            n_blocks = len(bh["blocks"])
            block_sizes = bh["block_sizes"]
            print(
                f"  {sid[:50]}: {n_blocks} blocks, "
                f"sizes: min={block_sizes.min()}, max={block_sizes.max()}, "
                f"mean={block_sizes.mean():.1f}"
            )

        print(f"\n  Total blocks across all sessions: {total_blocks}")

        # --- Within-block analysis ---
        # Aggregate to per-block mean SNR (across channels) to handle
        # sessions with different channel counts.
        first_mean_per_block_list = []
        second_mean_per_block_list = []
        first_ch_accum = []
        second_ch_accum = []
        for bh in all_block_halves:
            valid = ~np.isnan(bh["first_half_snr"]).any(axis=1)
            if valid.any():
                first_mean_per_block_list.extend(
                    np.nanmean(bh["first_half_snr"][valid], axis=1).tolist()
                )
                second_mean_per_block_list.extend(
                    np.nanmean(bh["second_half_snr"][valid], axis=1).tolist()
                )
                first_ch_accum.append(
                    np.nanmean(bh["first_half_snr"][valid], axis=0)
                )
                second_ch_accum.append(
                    np.nanmean(bh["second_half_snr"][valid], axis=0)
                )

        if first_mean_per_block_list:
            first_mean_per_block = np.array(first_mean_per_block_list)
            second_mean_per_block = np.array(second_mean_per_block_list)

            print(
                f"\n  Within-block analysis ({len(first_mean_per_block)} valid blocks):"
            )
            print(
                f"    Mean SNR first half:  {first_mean_per_block.mean():.6f}"
            )
            print(
                f"    Mean SNR second half: {second_mean_per_block.mean():.6f}"
            )
            ratio = first_mean_per_block.mean() / max(
                second_mean_per_block.mean(), 1e-12
            )
            print(f"    Ratio (first/second): {ratio:.3f}")

            pct_higher = (first_mean_per_block > second_mean_per_block).sum()
            print(
                f"    Blocks where first half > second half: "
                f"{pct_higher}/{len(first_mean_per_block)} "
                f"({100 * pct_higher / len(first_mean_per_block):.1f}%)"
            )

            try:
                stat, wp = _wil3(first_mean_per_block, second_mean_per_block)
                print(
                    f"    Wilcoxon first vs second half: stat={stat:.1f}, p={wp:.2e}"
                )
            except ValueError as e:
                print(
                    f"    Wilcoxon first vs second half: could not compute ({e})"
                )

            # Per-channel aggregated (per session, since channel counts differ)
            ch_higher_count = 0
            ch_total = 0
            for f_ch, s_ch in zip(first_ch_accum, second_ch_accum):
                ch_higher_count += int((f_ch > s_ch).sum())
                ch_total += len(f_ch)
            if ch_total > 0:
                print(
                    f"\n    Per-channel (across sessions): first half > second half in "
                    f"{ch_higher_count}/{ch_total} channels "
                    f"({100 * ch_higher_count / ch_total:.1f}%)"
                )

        # --- Across-block analysis ---
        print("\n  Across-block analysis:")
        all_corrs = []
        for bo in all_block_orders:
            snr_per_block = np.nanmean(bo["per_block_snr"], axis=1)
            valid = ~np.isnan(snr_per_block)
            if valid.sum() >= 3:
                rho, p = _sp4(bo["block_indices"][valid], snr_per_block[valid])
                all_corrs.append(rho)

        if all_corrs:
            corrs = np.array(all_corrs)
            print(f"    Sessions with ≥3 blocks: {len(corrs)}")
            print(f"    Mean Spearman(block_idx, SNR): {corrs.mean():.4f}")
            print(f"    Median Spearman: {np.median(corrs):.4f}")
            neg = (corrs < 0).sum()
            print(
                f"    Sessions with negative correlation: "
                f"{neg}/{len(corrs)} ({100 * neg / len(corrs):.1f}%)"
            )

        # --- Thirds analysis ---
        thirds_snr = [[], [], []]
        for bo in all_block_orders:
            snr = bo["per_block_snr"]
            n = len(snr)
            if n < 3:
                continue
            t1 = n // 3
            t2 = 2 * n // 3
            thirds_snr[0].append(np.nanmean(snr[:t1]))
            thirds_snr[1].append(np.nanmean(snr[t1:t2]))
            thirds_snr[2].append(np.nanmean(snr[t2:]))

        if all(len(t) > 0 for t in thirds_snr):
            print("\n  Block-thirds analysis:")
            labels = ["Early", "Middle", "Late"]
            for i, (label, vals) in enumerate(zip(labels, thirds_snr)):
                print(
                    f"    {label} blocks: mean SNR = {np.mean(vals):.6f} (n={len(vals)} sessions)"
                )

            try:
                stat, wp = _wil3(thirds_snr[0], thirds_snr[2])
                print(
                    f"    Wilcoxon early vs late blocks: stat={stat:.1f}, p={wp:.2e}"
                )
            except ValueError as e:
                print(
                    f"    Wilcoxon early vs late blocks: could not compute ({e})"
                )

        # --- Plots ---
        plot_within_block_habituation(
            all_block_halves,
            args.output_dir / "block_within_habituation.png",
        )
        plot_across_block_habituation(
            all_block_orders,
            args.output_dir / "block_across_habituation.png",
        )
        plot_block_erp_comparison(
            bb_epochs_dict,
            sfreq_dict,
            args.output_dir / "block_erp_comparison.png",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
