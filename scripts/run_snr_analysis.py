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
    build_channel_table,
    build_session_table,
    compute_channel_snr,
    compute_erp,
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
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
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
    """
    print(f"  Loading {h5_path.name} …")
    session = load_session(h5_path)

    epochs = extract_epochs(session)
    epochs = baseline_correct(epochs)

    bb_epochs = apply_broadband_filter(epochs, session.sampling_rate)
    hg_epochs = apply_high_gamma_filter(epochs, session.sampling_rate)

    ch_table = build_channel_table(
        session,
        bb_epochs,
        hg_epochs,
        snr_threshold=snr_threshold,
    )
    ses_table = build_session_table(ch_table)

    bb_snr = compute_channel_snr(bb_epochs)
    erp = compute_erp(bb_epochs)

    return ch_table, ses_table, bb_snr, erp


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

    for h5_path in h5_files:
        try:
            ch_table, ses_table, bb_snr, erp = analyse_session(
                h5_path,
                snr_threshold=args.snr_threshold,
            )
        except Exception as exc:
            print(f"  ⚠ Skipping {h5_path.name}: {exc}")
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

        session_info = load_session(h5_path)
        sfreq_dict[sid] = session_info.sampling_rate

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

    print("\nDone.")


if __name__ == "__main__":
    main()
