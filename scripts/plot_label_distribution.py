"""Plot label distribution across train/valid/test splits for a given
split type, fold, and task.

Usage:
    # Single plot (original behavior)
    uv run python scripts/plot_label_distribution.py \
        --dataset minipigs \
        --split-type intersubject \
        --fold 0 \
        --task acoustic_stim \
        --root data/processed

    # Batch analysis with remapping comparison (writes JSON + plots)
    uv run python scripts/plot_label_distribution.py \
        --dataset monkeys \
        --task acoustic_stim \
        --root data/processed \
        --batch-all \
        --output-dir reports/label_analysis
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from auditorydecoding.data.neurosoft_dataset import (
    NeurosoftMinipigs2026,
    NeurosoftMonkeys2026,
)


DATASET_CLASSES = {
    "minipigs": NeurosoftMinipigs2026,
    "monkeys": NeurosoftMonkeys2026,
}

CLASS_MAPPING = {
    "stim_100Hz": "low_bass",
    "stim_200Hz": "low_bass",
    "stim_300Hz": "low_bass",
    "stim_400Hz": "low_bass",
    "stim_500Hz": "low_bass",
    "stim_650Hz": "low_bass",
    "stim_800Hz": "mid_bass",
    "stim_1000Hz": "low_mids",
    "stim_1500Hz": "low_mids",
    "stim_2000Hz": "midrange",
    "stim_3000Hz": "midrange",
    "stim_4000Hz": "midrange",
    "stim_5000Hz": "high_mids",
    "stim_7700Hz": "low_treble",
    "stim_8000Hz": "low_treble",
    "stim_9500Hz": "low_treble",
    "stim_10000Hz": "mid_treble",
    "stim_12000Hz": "high_treble",
    "stim_13000Hz": "high_treble",
    "stim_15000Hz": "high_treble",
    "stim_16000Hz": "high_treble",
    "stim_18000Hz": "high_treble",
    "stim_20000Hz": "high_treble",
    "stim_30000Hz": "high_treble",
    "stim_40000Hz": "high_treble",
}

CLASS_ORDER = [
    "low_bass",
    "mid_bass",
    "low_mids",
    "midrange",
    "high_mids",
    "low_treble",
    "mid_treble",
    "high_treble",
]

SPLIT_CONFIGS = {
    "intersubject": {"folds": list(range(5)), "needs_fold": True},
    "intersession": {"folds": [0], "needs_fold": False},
    "intrasession-block": {"folds": list(range(3)), "needs_fold": True},
    "intrasession-causal": {"folds": [None], "needs_fold": False},
}

EXCLUDE_LABELS = {"stim_wn"}

RECORDING_IDS = [
    "sub-01_ses-01_task-AcousStim_acq-LH_desc-raw",
    "sub-01_ses-02_task-AcousStim_acq-LH_desc-raw",
    "sub-02_ses-01_task-AcousStim_acq-LH_desc-raw",
    "sub-02_ses-01_task-AcousStim_acq-RH_desc-raw",
    "sub-02_ses-02_task-AcousStim_acq-LH_desc-raw",
    "sub-02_ses-02_task-AcousStim_acq-RH_desc-raw",
    "sub-03_ses-01_task-AcousStim_acq-LH_desc-raw",
    "sub-03_ses-01_task-AcousStim_acq-RH_desc-raw",
    # "sub-03_ses-03_task-AcousStim_acq-LHanest_desc-raw",
    # "sub-03_ses-03_task-AcousStim_acq-RHanest_desc-raw",
    # "sub-03_ses-04_task-AcousStim_acq-LHanest_desc-raw",
    # "sub-03_ses-04_task-AcousStim_acq-RHanest_desc-raw",
    "sub-03_ses-06_task-AcousStim_acq-LH_desc-raw",
    "sub-03_ses-06_task-AcousStim_acq-RH_desc-raw",
    # "sub-03_ses-07_task-AcousStim_acq-LHanest_desc-raw",
    "sub-03_ses-07_task-AcousStim_acq-RH_desc-raw",
    # "sub-03_ses-07_task-AcousStim_acq-RHanest_desc-raw",
    "sub-04_ses-01_task-AcousStim_acq-LH_desc-raw",
    "sub-04_ses-01_task-AcousStim_acq-RH_desc-raw",
    "sub-04_ses-02_task-AcousStim_acq-LH_desc-raw",
    "sub-04_ses-02_task-AcousStim_acq-RH_desc-raw",
    "sub-04_ses-03_task-AcousStim_acq-LH_desc-raw",
    "sub-04_ses-03_task-AcousStim_acq-RH_desc-raw",
    "sub-04_ses-04_task-AcousStim_acq-LH_desc-raw",
    "sub-04_ses-04_task-AcousStim_acq-RH_desc-raw",
    "sub-05_ses-01_task-AcousStim_acq-LH_desc-raw",
    "sub-05_ses-01_task-AcousStim_acq-RH_desc-raw",
    "sub-05_ses-02_task-AcousStim_acq-LH_desc-raw",
    "sub-05_ses-02_task-AcousStim_acq-RH_desc-raw",
    "sub-06_ses-02_task-AcousStim_acq-LH_desc-raw",
    "sub-06_ses-02_task-AcousStim_acq-RH_desc-raw",
    "sub-07_ses-01_task-AcousStim_acq-LH_desc-raw",
    "sub-07_ses-02_task-AcousStim_acq-LH_desc-raw",
    "sub-07_ses-03_task-AcousStim_acq-LH_desc-raw",
    "sub-07_ses-03_task-AcousStim_acq-RH_desc-raw",
    "sub-07_ses-04_task-AcousStim_acq-LH_desc-raw",
    "sub-07_ses-04_task-AcousStim_acq-RH_desc-raw",
    "sub-07_ses-05_task-AcousStim_acq-LH_desc-raw",
    "sub-07_ses-05_task-AcousStim_acq-RH_desc-raw",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot label distribution per split."
    )
    parser.add_argument(
        "--dataset",
        choices=["minipigs", "monkeys"],
        required=True,
        help="Which dataset to use.",
    )
    parser.add_argument(
        "--split-type",
        choices=[
            "intersubject",
            "intersession",
            "intrasession-block",
            "intrasession-causal",
        ],
        default=None,
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold number (required for intersubject/intrasession-block).",
    )
    parser.add_argument(
        "--task",
        choices=["on_vs_off", "acoustic_stim"],
        default="acoustic_stim",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/processed",
        help="Root directory containing processed HDF5 files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the figure (single mode).",
    )
    parser.add_argument(
        "--remap",
        action="store_true",
        help="Apply frequency-band class remapping.",
    )
    parser.add_argument(
        "--batch-all",
        action="store_true",
        help="Run analysis for all split types and folds, save JSON + plots.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/label_analysis",
        help="Output directory for batch mode.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def get_label_counts_per_split(
    dataset, splits: list[str], remap: bool = False
) -> dict[str, Counter]:
    """Return {split_name: Counter(label -> count)} for each split."""
    counts: dict[str, Counter] = {}
    for split in splits:
        intervals = dataset.get_sampling_intervals(split)
        counter: Counter = Counter()
        for rid, interval in intervals.items():
            if len(interval) == 0:
                continue
            if hasattr(interval, "behavior_labels"):
                labels = interval.behavior_labels
                for label in labels:
                    lbl = str(label)
                    if lbl in EXCLUDE_LABELS:
                        continue
                    if remap and lbl in CLASS_MAPPING:
                        lbl = CLASS_MAPPING[lbl]
                    elif remap and lbl not in CLASS_MAPPING:
                        continue
                    counter[lbl] += 1
        counts[split] = counter
    return counts


def get_subject_info_per_split(
    dataset, splits: list[str]
) -> dict[str, set[str]]:
    """Return {split_name: set of subject IDs with non-empty intervals}."""
    subjects: dict[str, set[str]] = {}
    for split in splits:
        intervals = dataset.get_sampling_intervals(split)
        split_subjects: set[str] = set()
        for rid, interval in intervals.items():
            if len(interval) == 0:
                continue
            data = dataset.get_recording(rid)
            split_subjects.add(str(data.subject.id))
        subjects[split] = split_subjects
    return subjects


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sort_labels(labels: set[str], remap: bool = False) -> list[str]:
    if remap:
        return [label for label in CLASS_ORDER if label in labels]

    def _sort_key(label: str):
        m = re.search(r"(\d+)", label)
        if m:
            return (1, int(m.group(1)))
        return (0, label)

    return sorted(labels, key=_sort_key)


def compute_imbalance_metrics(counts: Counter) -> dict:
    if not counts:
        return {
            "max_min_ratio": 0,
            "cv": 0,
            "max_class": "",
            "min_class": "",
            "max_count": 0,
            "min_count": 0,
            "mean_count": 0,
            "total": 0,
            "n_classes": 0,
        }
    vals = list(counts.values())
    max_val = max(vals)
    min_val = min(vals)
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    max_class = max(counts, key=counts.get)
    min_class = min(counts, key=counts.get)
    return {
        "max_min_ratio": max_val / min_val if min_val > 0 else float("inf"),
        "cv": float(std_val / mean_val) if mean_val > 0 else 0,
        "max_class": max_class,
        "max_count": max_val,
        "min_class": min_class,
        "min_count": min_val,
        "mean_count": float(mean_val),
        "total": sum(vals),
        "n_classes": len(vals),
    }


def compute_split_proportion_stability(
    counts: dict[str, Counter],
) -> dict:
    all_labels = set()
    for c in counts.values():
        all_labels.update(c.keys())

    proportions = {}
    for split, counter in counts.items():
        total = sum(counter.values())
        if total == 0:
            continue
        proportions[split] = {
            lbl: counter.get(lbl, 0) / total for lbl in all_labels
        }

    if len(proportions) < 2:
        return {}

    label_std = {}
    for lbl in all_labels:
        vals = [proportions[s].get(lbl, 0) for s in proportions]
        label_std[lbl] = {
            "proportions": {
                s: round(proportions[s].get(lbl, 0) * 100, 2)
                for s in proportions
            },
            "std": round(float(np.std(vals)) * 100, 2),
            "max_diff": round(float(max(vals) - min(vals)) * 100, 2),
        }

    return label_std


# ---------------------------------------------------------------------------
# Bar-chart plots (existing)
# ---------------------------------------------------------------------------


def plot_distributions(
    counts: dict[str, Counter],
    subjects: dict[str, set[str]],
    title: str,
    output: str | None,
    remap: bool = False,
) -> None:
    all_labels = _sort_labels(
        set().union(*(c.keys() for c in counts.values())), remap=remap
    )
    splits = list(counts.keys())

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(10, len(all_labels) * 0.8), 10),
        height_ratios=[3, 1],
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    ax = axes[0]
    x = np.arange(len(all_labels))
    n_splits = len(splits)
    width = 0.8 / n_splits
    colors = {"train": "#2196F3", "valid": "#FF9800", "test": "#4CAF50"}

    for i, split in enumerate(splits):
        vals = [counts[split].get(label, 0) for label in all_labels]
        offset = (i - n_splits / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=f"{split} ({sum(vals)} trials)",
            color=colors.get(split, f"C{i}"),
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    str(val),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of trials")
    ax.set_title("Label counts per split")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax2 = axes[1]
    for i, split in enumerate(splits):
        total = sum(counts[split].values())
        if total == 0:
            continue
        vals = [
            counts[split].get(label, 0) / total * 100 for label in all_labels
        ]
        offset = (i - n_splits / 2 + 0.5) * width
        ax2.bar(
            x + offset,
            vals,
            width,
            label=split,
            color=colors.get(split, f"C{i}"),
            edgecolor="white",
            linewidth=0.5,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Proportion (%)")
    ax2.set_title("Label proportion within each split")
    ax2.legend(loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    subject_text = " | ".join(
        f"{split}: {', '.join(sorted(subjects[split])) if subjects[split] else '(none)'}"
        for split in splits
    )
    fig.text(
        0.5,
        0.01,
        f"Subjects — {subject_text}",
        ha="center",
        fontsize=9,
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {output}")
    else:
        plt.show()
    plt.close(fig)


def plot_before_after_comparison(
    counts_orig: dict[str, Counter],
    counts_remap: dict[str, Counter],
    title: str,
    output: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    colors = {"train": "#2196F3", "valid": "#FF9800", "test": "#4CAF50"}
    splits = list(counts_orig.keys())

    for col, (counts, remap, subtitle) in enumerate([
        (counts_orig, False, "Original (per-frequency)"),
        (counts_remap, True, "Remapped (frequency bands)"),
    ]):
        all_labels = _sort_labels(
            set().union(*(c.keys() for c in counts.values())), remap=remap
        )
        x = np.arange(len(all_labels))
        n_splits = len(splits)
        width = 0.8 / n_splits

        ax = axes[0, col]
        for i, split in enumerate(splits):
            vals = [counts[split].get(label, 0) for label in all_labels]
            offset = (i - n_splits / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=f"{split} ({sum(vals)})",
                color=colors.get(split, f"C{i}"),
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        str(val),
                        ha="center",
                        va="bottom",
                        fontsize=6,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Count")
        ax.set_title(f"{subtitle} — Counts")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        ax2 = axes[1, col]
        for i, split in enumerate(splits):
            total = sum(counts[split].values())
            if total == 0:
                continue
            vals = [
                counts[split].get(label, 0) / total * 100
                for label in all_labels
            ]
            offset = (i - n_splits / 2 + 0.5) * width
            ax2.bar(
                x + offset,
                vals,
                width,
                label=split,
                color=colors.get(split, f"C{i}"),
                edgecolor="white",
                linewidth=0.5,
            )
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel("Proportion (%)")
        ax2.set_title(f"{subtitle} — Proportions")
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.3)
        ax2.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Comparison figure saved to {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Heatmap plots (new)
# ---------------------------------------------------------------------------


def _diverging_cmap():
    return plt.cm.RdYlGn_r


def _sequential_cmap():
    return plt.cm.YlOrRd


def plot_heatmap_imbalance_overview(
    all_results: list[dict],
    dataset_name: str,
    output: str,
) -> None:
    """Q1 heatmap: class proportions (%) before vs after remapping.

    Uses intrasession-block fold 0 as a clean reference (no split-assignment
    confounds).  Two side-by-side heatmaps with annotated deviation from
    uniform.
    """
    ref = None
    for r in all_results:
        if r["split_type"] == "intrasession-block" and r.get("fold") == 0:
            ref = r
            break
    if ref is None:
        ref = all_results[0]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), width_ratios=[2.8, 1])
    fig.suptitle(
        f"{dataset_name.upper()} — Class proportion (%) by split: "
        f"original vs remapped",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    splits = ["train", "valid", "test"]

    for ax_idx, (mode, label_order_fn) in enumerate([
        (
            "original",
            lambda r: _sort_labels(
                set().union(*(r[mode][s]["counts"].keys() for s in splits)),
                remap=False,
            ),
        ),
        ("remapped", lambda r: CLASS_ORDER),
    ]):
        ax = axes[ax_idx]
        labels = label_order_fn(ref)
        n_labels = len(labels)
        n_splits = len(splits)
        matrix = np.zeros((n_labels, n_splits))
        uniform = 100.0 / n_labels if n_labels > 0 else 0

        for j, s in enumerate(splits):
            total = sum(ref[mode][s]["counts"].values())
            for i, lbl in enumerate(labels):
                cnt = ref[mode][s]["counts"].get(lbl, 0)
                matrix[i, j] = cnt / total * 100 if total > 0 else 0

        vmax = max(matrix.max(), uniform * 2) if n_labels > 0 else 1
        ax.imshow(
            matrix,
            aspect="auto",
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
        )
        ax.set_xticks(range(n_splits))
        ax.set_xticklabels(splits, fontsize=10)
        ax.set_yticks(range(n_labels))
        ax.set_yticklabels(labels, fontsize=9)

        for i in range(n_labels):
            for j in range(n_splits):
                val = matrix[i, j]
                color = "white" if val > vmax * 0.6 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                    fontweight="bold" if abs(val - uniform) > 3 else "normal",
                )

        ax.axhline(y=-0.5, color="gray", linewidth=0.5)
        for i in range(n_labels):
            ax.axhline(y=i + 0.5, color="gray", linewidth=0.5)

        subtitle = (
            "Original (per-frequency)"
            if mode == "original"
            else f"Remapped ({len(CLASS_ORDER)} bands)"
        )
        n_cls = n_labels
        ax.set_title(
            f"{subtitle}\n{n_cls} classes · uniform = {uniform:.1f}%",
            fontsize=11,
        )

    fig.colorbar(
        axes[1].images[0],
        ax=axes,
        shrink=0.8,
        label="Proportion (%)",
    )
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Heatmap (imbalance overview) saved to {output}")
    plt.close(fig)


def plot_heatmap_cross_config_proportions(
    all_results: list[dict],
    dataset_name: str,
    output: str,
) -> None:
    """Q2 heatmap: remapped band proportions across every (split_type, fold,
    split) configuration.

    Rows = configuration, columns = bands.  Colour = proportion (%).
    Allows visual inspection of whether proportions are stable.
    """
    bands = CLASS_ORDER
    splits = ["train", "valid", "test"]
    uniform = 100.0 / len(bands)

    rows = []
    row_labels = []

    for r in all_results:
        st = r["split_type"]
        fold = r.get("fold")
        fold_str = f"f{fold}" if fold is not None else ""
        for s in splits:
            total = sum(r["remapped"][s]["counts"].values())
            if total == 0:
                continue
            row = []
            for b in bands:
                cnt = r["remapped"][s]["counts"].get(b, 0)
                row.append(cnt / total * 100)
            rows.append(row)
            row_labels.append(f"{st} {fold_str} {s}")

    if not rows:
        return

    matrix = np.array(rows)
    n_rows, n_cols = matrix.shape

    fig, ax = plt.subplots(figsize=(10, max(6, n_rows * 0.35)))
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=max(matrix.max(), uniform * 2.5),
    )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(bands, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)

    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            vmax_local = max(matrix.max(), 1)
            color = "white" if val > vmax_local * 0.6 else "black"
            ax.text(
                j,
                i,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )

    # Horizontal separators between split types
    prev_st = None
    for idx, lbl in enumerate(row_labels):
        st = lbl.rsplit(" ", 1)[0].rsplit(" ", 1)[0]
        if prev_st is not None and st != prev_st:
            ax.axhline(y=idx - 0.5, color="black", linewidth=1.5)
        prev_st = st

    ax.set_title(
        f"{dataset_name.upper()} — Remapped band proportions (%) across all configurations\n"
        f"Uniform = {uniform:.1f}% per band",
        fontsize=13,
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, shrink=0.6, label="Proportion (%)")
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Heatmap (cross-config proportions) saved to {output}")
    plt.close(fig)


def plot_heatmap_trainval_vs_test_drift(
    all_results: list[dict],
    dataset_name: str,
    output: str,
) -> None:
    """Q2 heatmap: proportion drift between (train+valid) and test for each
    band, across all (split_type, fold) combinations.

    Colour = |proportion_trainval - proportion_test| in percentage points.
    Green = close to zero (good).  Red = large drift (bad).
    """
    bands = CLASS_ORDER

    rows = []
    row_labels = []

    for r in all_results:
        st = r["split_type"]
        fold = r.get("fold")
        fold_str = f"f{fold}" if fold is not None else ""

        stability = r["remapped"].get("trainval_test_stability", {})
        if not stability:
            continue

        row = []
        for b in bands:
            info = stability.get(b, {})
            row.append(info.get("max_diff", 0))
        rows.append(row)
        row_labels.append(f"{st} {fold_str}")

    if not rows:
        return

    matrix = np.array(rows)

    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.45)))
    vmax = max(matrix.max(), 5)
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=vmax,
    )

    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(bands, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(rows)):
        for j in range(len(bands)):
            val = matrix[i, j]
            color = "white" if val > vmax * 0.55 else "black"
            ax.text(
                j,
                i,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color=color,
                fontweight="bold" if val > 5 else "normal",
            )

    ax.set_title(
        f"{dataset_name.upper()} — (Train+Valid) vs Test proportion drift (pp)\n"
        f"Lower is better · values > 5pp bolded",
        fontsize=13,
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, shrink=0.7, label="Absolute drift (pp)")
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Heatmap (TV vs test drift) saved to {output}")
    plt.close(fig)


def plot_heatmap_train_valid_test_drift(
    all_results: list[dict],
    dataset_name: str,
    output: str,
) -> None:
    """Q2 heatmap: max proportion drift across train / valid / test
    individually for each band across all configurations.

    Colour = max_diff across the three splits in percentage points.
    """
    bands = CLASS_ORDER

    rows = []
    row_labels = []

    for r in all_results:
        st = r["split_type"]
        fold = r.get("fold")
        fold_str = f"f{fold}" if fold is not None else ""

        stability = r["remapped"].get("proportion_stability", {})
        if not stability:
            continue

        row = []
        for b in bands:
            info = stability.get(b, {})
            row.append(info.get("max_diff", 0))
        rows.append(row)
        row_labels.append(f"{st} {fold_str}")

    if not rows:
        return

    matrix = np.array(rows)

    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.45)))
    vmax = max(matrix.max(), 5)
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=vmax,
    )

    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(bands, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(rows)):
        for j in range(len(bands)):
            val = matrix[i, j]
            color = "white" if val > vmax * 0.55 else "black"
            ax.text(
                j,
                i,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color=color,
                fontweight="bold" if val > 5 else "normal",
            )

    ax.set_title(
        f"{dataset_name.upper()} — Train vs Valid vs Test max proportion drift (pp)\n"
        f"Lower is better · values > 5pp bolded",
        fontsize=13,
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, shrink=0.7, label="Max drift (pp)")
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Heatmap (train/valid/test drift) saved to {output}")
    plt.close(fig)


def plot_heatmap_maxmin_ratio(
    all_results: list[dict],
    dataset_name: str,
    output: str,
) -> None:
    """Q1 heatmap: max/min class ratio across all configurations and splits.

    Rows = (split_type, fold), columns = (original_train, original_valid,
    original_test, remapped_train, remapped_valid, remapped_test).
    """
    splits = ["train", "valid", "test"]
    col_labels = [f"orig {s}" for s in splits] + [f"remap {s}" for s in splits]

    rows = []
    row_labels = []

    for r in all_results:
        st = r["split_type"]
        fold = r.get("fold")
        fold_str = f"f{fold}" if fold is not None else ""
        row = []
        for mode in ["original", "remapped"]:
            for s in splits:
                ratio = r[mode][s]["imbalance"].get("max_min_ratio", 0)
                if ratio == float("inf"):
                    ratio = 0
                row.append(ratio)
        rows.append(row)
        row_labels.append(f"{st} {fold_str}")

    if not rows:
        return

    matrix = np.array(rows)

    fig, ax = plt.subplots(figsize=(12, max(4, len(rows) * 0.45)))
    vmax = min(matrix.max(), 50)
    im = ax.imshow(
        np.clip(matrix, 0, vmax),
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=1,
        vmax=vmax,
    )

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Vertical separator between original and remapped
    ax.axvline(x=2.5, color="black", linewidth=2)

    for i in range(len(rows)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            display = f"{val:.1f}" if val < 100 else f"{val:.0f}"
            if val == 0:
                display = "—"
            color = "white" if val > vmax * 0.55 else "black"
            ax.text(
                j,
                i,
                display,
                ha="center",
                va="center",
                fontsize=8,
                color=color,
                fontweight="bold" if val > 5 else "normal",
            )

    ax.set_title(
        f"{dataset_name.upper()} — Max/Min class ratio across all configurations\n"
        f"Left: original · Right: remapped · Lower is better (1.0 = perfect)",
        fontsize=13,
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, shrink=0.7, label="Max/Min ratio")
    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Heatmap (max/min ratio) saved to {output}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis core
# ---------------------------------------------------------------------------


def analyze_single(
    dataset_name: str,
    split_type: str,
    fold: int | None,
    task: str,
    root: str,
) -> dict:
    dataset_cls = DATASET_CLASSES[dataset_name]
    dataset = dataset_cls(
        root=root,
        split_type=split_type,
        fold_num=fold,
        task_type=task,
        recording_ids=RECORDING_IDS,
    )

    splits = ["train", "valid", "test"]
    counts_orig = get_label_counts_per_split(dataset, splits, remap=False)
    counts_remap = get_label_counts_per_split(dataset, splits, remap=True)
    subjects = get_subject_info_per_split(dataset, splits)

    result = {
        "dataset": dataset_name,
        "split_type": split_type,
        "fold": fold,
        "task": task,
        "n_recordings": len(dataset.recording_ids),
        "subjects": {s: sorted(subjects[s]) for s in splits},
        "original": {},
        "remapped": {},
    }

    for split in splits:
        result["original"][split] = {
            "counts": dict(counts_orig[split]),
            "imbalance": compute_imbalance_metrics(counts_orig[split]),
        }
        result["remapped"][split] = {
            "counts": dict(counts_remap[split]),
            "imbalance": compute_imbalance_metrics(counts_remap[split]),
        }

    result["original"]["proportion_stability"] = (
        compute_split_proportion_stability(counts_orig)
    )
    result["remapped"]["proportion_stability"] = (
        compute_split_proportion_stability(counts_remap)
    )

    trainval_orig = counts_orig["train"] + counts_orig["valid"]
    trainval_remap = counts_remap["train"] + counts_remap["valid"]
    result["original"]["trainval_vs_test"] = {
        "trainval": compute_imbalance_metrics(trainval_orig),
        "test": compute_imbalance_metrics(counts_orig["test"]),
    }
    result["remapped"]["trainval_vs_test"] = {
        "trainval": compute_imbalance_metrics(trainval_remap),
        "test": compute_imbalance_metrics(counts_remap["test"]),
    }

    result["original"]["trainval_test_stability"] = (
        compute_split_proportion_stability({
            "train+valid": trainval_orig,
            "test": counts_orig["test"],
        })
    )
    result["remapped"]["trainval_test_stability"] = (
        compute_split_proportion_stability({
            "train+valid": trainval_remap,
            "test": counts_remap["test"],
        })
    )

    return result, counts_orig, counts_remap, subjects


def batch_analysis(dataset_name: str, task: str, root: str, output_dir: str):
    out = Path(output_dir) / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    all_results = []

    for split_type, config in SPLIT_CONFIGS.items():
        for fold in config["folds"]:
            fold_arg = fold if config["needs_fold"] else fold
            fold_label = f"fold{fold}" if fold is not None else "fold0"
            print(f"\n{'=' * 60}")
            print(f"  {dataset_name} / {split_type} / {fold_label} / {task}")
            print(f"{'=' * 60}")

            try:
                result, counts_orig, counts_remap, subjects = analyze_single(
                    dataset_name,
                    split_type,
                    fold_arg,
                    task,
                    root,
                )
                all_results.append(result)

                plot_before_after_comparison(
                    counts_orig,
                    counts_remap,
                    title=f"{dataset_name} / {split_type} / {fold_label} / {task}",
                    output=str(
                        out / f"{split_type}_{fold_label}_comparison.png"
                    ),
                )

                for remap in [False, True]:
                    counts = counts_remap if remap else counts_orig
                    tag = "remapped" if remap else "original"
                    plot_distributions(
                        counts,
                        subjects,
                        title=(
                            f"{dataset_name} / {split_type} / {fold_label}"
                            f" / {task} ({tag})"
                        ),
                        output=str(
                            out / f"{split_type}_{fold_label}_{tag}.png"
                        ),
                        remap=remap,
                    )

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback

                traceback.print_exc()
                all_results.append({
                    "dataset": dataset_name,
                    "split_type": split_type,
                    "fold": fold,
                    "task": task,
                    "error": str(e),
                })

    # Filter to successful results for heatmaps
    ok_results = [r for r in all_results if "error" not in r]

    if ok_results:
        print("\nGenerating heatmap plots...")
        plot_heatmap_imbalance_overview(
            ok_results,
            dataset_name,
            str(out / "heatmap_imbalance_overview.png"),
        )
        plot_heatmap_cross_config_proportions(
            ok_results,
            dataset_name,
            str(out / "heatmap_cross_config_proportions.png"),
        )
        plot_heatmap_trainval_vs_test_drift(
            ok_results,
            dataset_name,
            str(out / "heatmap_trainval_vs_test_drift.png"),
        )
        plot_heatmap_train_valid_test_drift(
            ok_results,
            dataset_name,
            str(out / "heatmap_train_valid_test_drift.png"),
        )
        plot_heatmap_maxmin_ratio(
            ok_results,
            dataset_name,
            str(out / "heatmap_maxmin_ratio.png"),
        )

    json_path = out / "analysis_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    return all_results


def main() -> None:
    args = parse_args()

    if args.batch_all:
        batch_analysis(args.dataset, args.task, args.root, args.output_dir)
        return

    if args.split_type is None:
        raise ValueError(
            "--split-type is required in single mode (or use --batch-all)"
        )

    dataset_cls = DATASET_CLASSES[args.dataset]
    dataset = dataset_cls(
        root=args.root,
        split_type=args.split_type,
        fold_num=args.fold,
        task_type=args.task,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Split type: {args.split_type}")
    print(f"Fold: {args.fold}")
    print(f"Task: {args.task}")
    print(f"Remap: {args.remap}")
    print(f"Recordings: {len(dataset.recording_ids)}")
    print()

    splits = ["train", "valid", "test"]
    counts = get_label_counts_per_split(dataset, splits, remap=args.remap)
    subjects = get_subject_info_per_split(dataset, splits)

    for split in splits:
        print(f"\n--- {split.upper()} ---")
        print(f"  Subjects: {sorted(subjects[split])}")
        total = sum(counts[split].values())
        print(f"  Total trials: {total}")
        for label in _sort_labels(set(counts[split].keys()), remap=args.remap):
            c = counts[split][label]
            pct = c / total * 100 if total > 0 else 0
            print(f"    {label}: {c} ({pct:.1f}%)")

        metrics = compute_imbalance_metrics(counts[split])
        print(
            f"  Imbalance: max/min={metrics['max_min_ratio']:.2f}, "
            f"CV={metrics['cv']:.3f}"
        )

    fold_str = f"fold {args.fold}" if args.fold is not None else "fold 0"
    remap_str = " (remapped)" if args.remap else ""
    title = (
        f"Label Distribution — {args.dataset} / {args.split_type} / "
        f"{fold_str} / {args.task}{remap_str}"
    )

    plot_distributions(counts, subjects, title, args.output, remap=args.remap)


if __name__ == "__main__":
    main()
