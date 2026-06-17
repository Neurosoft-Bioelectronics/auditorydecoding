"""Plot label distribution across train/valid/test splits for a given
split type, fold, and task.

Usage:
    uv run python scripts/plot_label_distribution.py \
        --dataset minipigs \
        --split-type intersubject \
        --fold 0 \
        --task on_vs_off \
        --root data/processed

    uv run python scripts/plot_label_distribution.py \
        --dataset monkeys \
        --split-type intersession \
        --task acoustic_stim \
        --root data/processed
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

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
        required=True,
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
        default="on_vs_off",
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
        help="Path to save the figure. If not provided, displays interactively.",
    )
    return parser.parse_args()


def get_label_counts_per_split(
    dataset, splits: list[str]
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
                    counter[str(label)] += 1
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


def _sort_labels(labels: set[str]) -> list[str]:
    """Sort labels with frequencies in ascending numeric order, 'stim_wn' last."""

    def _sort_key(label: str):
        if label == "stim_wn":
            return (1, float("inf"))
        m = re.search(r"(\d+)", label)
        if m:
            return (1, int(m.group(1)))
        return (0, label)

    return sorted(labels, key=_sort_key)


def plot_distributions(
    counts: dict[str, Counter],
    subjects: dict[str, set[str]],
    title: str,
    output: str | None,
) -> None:
    all_labels = _sort_labels(set().union(*(c.keys() for c in counts.values())))
    splits = list(counts.keys())

    fig, axes = plt.subplots(
        2, 1, figsize=(max(10, len(all_labels) * 0.8), 10), height_ratios=[3, 1]
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # --- Top: grouped bar chart of label counts per split ---
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

    # --- Bottom: proportion within each split ---
    ax2 = axes[1]
    bottom_vals = {split: np.zeros(len(all_labels)) for split in splits}
    for split in splits:
        total = sum(counts[split].values())
        if total == 0:
            continue
        for j, label in enumerate(all_labels):
            bottom_vals[split][j] = counts[split].get(label, 0) / total * 100

    for i, split in enumerate(splits):
        offset = (i - n_splits / 2 + 0.5) * width
        ax2.bar(
            x + offset,
            bottom_vals[split],
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

    # Add subject info as text
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


def main() -> None:
    args = parse_args()

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
    print(f"Recordings: {len(dataset.recording_ids)}")
    print()

    splits = ["train", "valid", "test"]

    print("Collecting label distributions...")
    counts = get_label_counts_per_split(dataset, splits)
    subjects = get_subject_info_per_split(dataset, splits)

    for split in splits:
        print(f"\n--- {split.upper()} ---")
        print(f"  Subjects: {sorted(subjects[split])}")
        total = sum(counts[split].values())
        print(f"  Total trials: {total}")
        for label in sorted(counts[split].keys()):
            c = counts[split][label]
            pct = c / total * 100 if total > 0 else 0
            print(f"    {label}: {c} ({pct:.1f}%)")

    fold_str = f"fold {args.fold}" if args.fold is not None else "fold 0"
    title = (
        f"Label Distribution — {args.dataset} / {args.split_type} / "
        f"{fold_str} / {args.task}"
    )

    plot_distributions(counts, subjects, title, args.output)


if __name__ == "__main__":
    main()
