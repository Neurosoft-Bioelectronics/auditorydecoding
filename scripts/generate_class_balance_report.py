"""Generate a markdown report from the batch label analysis JSON files.

Usage:
    uv run python scripts/generate_class_balance_report.py \
        --input-dir reports/label_analysis \
        --output reports/label_analysis/class_balance_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CLASS_ORDER = ["sub_bass", "low", "low_mid", "mid", "high_mid", "high"]


def load_results(input_dir: str) -> dict[str, list[dict]]:
    results = {}
    for dataset in ["minipigs", "monkeys"]:
        path = Path(input_dir) / dataset / "analysis_results.json"
        if path.exists():
            with open(path) as f:
                results[dataset] = [r for r in json.load(f) if "error" not in r]
    return results


def fmt_pct(val: float) -> str:
    return f"{val:.1f}%"


def fmt_ratio(val: float) -> str:
    if val == float("inf"):
        return "∞"
    return f"{val:.2f}"


def make_counts_table(result: dict, mode: str = "original") -> str:
    data = result[mode]
    splits = ["train", "valid", "test"]

    if mode == "remapped":
        labels = CLASS_ORDER
    else:
        all_labels = set()
        for s in splits:
            all_labels.update(data[s]["counts"].keys())
        import re

        def sort_key(label: str):
            if label == "stim_wn":
                return (1, float("inf"))
            m = re.search(r"(\d+)", label)
            return (1, int(m.group(1))) if m else (0, label)

        labels = sorted(all_labels, key=sort_key)

    header = (
        "| Label | "
        + " | ".join(f"{s} (count) | {s} (%)" for s in splits)
        + " | Total |"
    )
    sep = "|" + "---|" * (len(splits) * 2 + 2)

    rows = []
    totals = {s: sum(data[s]["counts"].values()) for s in splits}
    grand_totals = {s: 0 for s in splits}

    for label in labels:
        row = f"| {label} |"
        label_total = 0
        for s in splits:
            count = data[s]["counts"].get(label, 0)
            pct = count / totals[s] * 100 if totals[s] > 0 else 0
            row += f" {count} | {pct:.1f}% |"
            label_total += count
            grand_totals[s] += count
        row += f" {label_total} |"
        rows.append(row)

    total_row = "| **Total** |"
    grand = 0
    for s in splits:
        total_row += f" **{grand_totals[s]}** | 100% |"
        grand += grand_totals[s]
    total_row += f" **{grand}** |"
    rows.append(total_row)

    return "\n".join([header, sep] + rows)


def make_imbalance_summary(result: dict, mode: str = "original") -> str:
    data = result[mode]
    splits = ["train", "valid", "test"]

    header = "| Metric | " + " | ".join(splits) + " |"
    sep = "|" + "---|" * (len(splits) + 1)

    rows = []
    for metric_name, metric_key in [
        ("Max/Min ratio", "max_min_ratio"),
        ("CV", "cv"),
        ("# Classes", "n_classes"),
        ("Total trials", "total"),
        ("Largest class", "max_class"),
        ("Largest count", "max_count"),
        ("Smallest class", "min_class"),
        ("Smallest count", "min_count"),
    ]:
        row = f"| {metric_name} |"
        for s in splits:
            val = data[s]["imbalance"].get(metric_key, "")
            if metric_key == "max_min_ratio":
                row += f" {fmt_ratio(val)} |"
            elif metric_key == "cv":
                row += f" {val:.3f} |"
            else:
                row += f" {val} |"
        rows.append(row)

    return "\n".join([header, sep] + rows)


def make_stability_table(result: dict, mode: str = "original") -> str:
    stability = result[mode].get("proportion_stability", {})
    if not stability:
        return "_No stability data available._"

    if mode == "remapped":
        labels = [label for label in CLASS_ORDER if label in stability]
    else:
        import re

        def sort_key(label: str):
            if label == "stim_wn":
                return (1, float("inf"))
            m = re.search(r"(\d+)", label)
            return (1, int(m.group(1))) if m else (0, label)

        labels = sorted(stability.keys(), key=sort_key)

    splits_in_data = ["train", "valid", "test"]
    header = (
        "| Label | "
        + " | ".join(f"{s} (%)" for s in splits_in_data)
        + " | Std (pp) | Max Diff (pp) |"
    )
    sep = "|" + "---|" * (len(splits_in_data) + 3)

    rows = []
    for label in labels:
        info = stability[label]
        row = f"| {label} |"
        for s in splits_in_data:
            row += f" {info['proportions'].get(s, 0):.2f} |"
        row += f" {info['std']:.2f} | {info['max_diff']:.2f} |"
        rows.append(row)

    return "\n".join([header, sep] + rows)


def make_trainval_test_stability_table(
    result: dict, mode: str = "original"
) -> str:
    stability = result[mode].get("trainval_test_stability", {})
    if not stability:
        return "_No stability data available._"

    if mode == "remapped":
        labels = [label for label in CLASS_ORDER if label in stability]
    else:
        import re

        def sort_key(label: str):
            if label == "stim_wn":
                return (1, float("inf"))
            m = re.search(r"(\d+)", label)
            return (1, int(m.group(1))) if m else (0, label)

        labels = sorted(stability.keys(), key=sort_key)

    header = "| Label | Train+Valid (%) | Test (%) | Std (pp) | Max Diff (pp) |"
    sep = "|---|---|---|---|---|"

    rows = []
    for label in labels:
        info = stability[label]
        tv = info["proportions"].get("train+valid", 0)
        te = info["proportions"].get("test", 0)
        row = f"| {label} | {tv:.2f} | {te:.2f} | {info['std']:.2f} | {info['max_diff']:.2f} |"
        rows.append(row)

    return "\n".join([header, sep] + rows)


def generate_report(all_results: dict[str, list[dict]], output: str):
    lines = []
    lines.append("# Class Balance Analysis Report: Frequency Band Remapping")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(
        "This report analyzes the class distribution of acoustic stimulation trials"
    )
    lines.append(
        "in the minipigs and monkeys datasets, comparing the original per-frequency"
    )
    lines.append(
        "labels (up to 26 classes) against the proposed frequency-band remapping (6 classes)."
    )
    lines.append("")
    lines.append("### Frequency Band Mapping")
    lines.append("")
    lines.append("| Band | Frequencies |")
    lines.append("|------|------------|")
    lines.append("| sub_bass | 100, 200, 300, 400, 500 Hz |")
    lines.append("| low | 650, 800, 1000, 1500 Hz |")
    lines.append("| low_mid | 2000, 3000, 4000 Hz |")
    lines.append("| mid | 5000, 7700, 8000, 9500 Hz |")
    lines.append("| high_mid | 10000, 12000, 13000, 15000, 16000 Hz |")
    lines.append("| high | 18000, 20000, 30000, 40000 Hz |")
    lines.append("")
    lines.append("### Key Questions")
    lines.append("")
    lines.append(
        "1. **How unbalanced are the classes before and after the remapping?**"
    )
    lines.append(
        "2. **Are these imbalances relatively constant across splits for all folds and split types?**"
    )
    lines.append("   - Balance between (train + val) and test")
    lines.append("   - Balance between train, val, and test individually")
    lines.append("")

    for dataset_name, results in all_results.items():
        lines.append("---")
        lines.append("")
        lines.append(f"## Dataset: {dataset_name.upper()}")
        lines.append("")

        # Group by split type
        by_split_type: dict[str, list[dict]] = {}
        for r in results:
            st = r["split_type"]
            by_split_type.setdefault(st, []).append(r)

        # First: global overview (aggregate across all data)
        first = results[0]
        lines.append(
            "### Global Class Distribution (intersubject fold 0 as reference)"
        )
        lines.append("")
        lines.append(f"**Subjects:** {first.get('subjects', {})}")
        lines.append("")

        lines.append("#### Original (per-frequency) Distribution")
        lines.append("")
        lines.append(make_counts_table(first, "original"))
        lines.append("")
        lines.append("**Imbalance Metrics:**")
        lines.append("")
        lines.append(make_imbalance_summary(first, "original"))
        lines.append("")

        lines.append("#### Remapped (frequency bands) Distribution")
        lines.append("")
        lines.append(make_counts_table(first, "remapped"))
        lines.append("")
        lines.append("**Imbalance Metrics:**")
        lines.append("")
        lines.append(make_imbalance_summary(first, "remapped"))
        lines.append("")

        # Compare imbalance improvement
        orig_ratios = {
            s: first["original"][s]["imbalance"]["max_min_ratio"]
            for s in ["train", "valid", "test"]
        }
        remap_ratios = {
            s: first["remapped"][s]["imbalance"]["max_min_ratio"]
            for s in ["train", "valid", "test"]
        }
        lines.append("#### Imbalance Improvement Summary (fold 0)")
        lines.append("")
        lines.append(
            "| Split | Original Max/Min | Remapped Max/Min | Improvement |"
        )
        lines.append("|---|---|---|---|")
        for s in ["train", "valid", "test"]:
            orig = orig_ratios[s]
            remap = remap_ratios[s]
            if orig != float("inf") and remap != float("inf") and remap > 0:
                improvement = f"{(1 - remap / orig) * 100:.1f}%"
            else:
                improvement = "N/A"
            lines.append(
                f"| {s} | {fmt_ratio(orig)} | {fmt_ratio(remap)} | {improvement} |"
            )
        lines.append("")

        # Per split type analysis
        for split_type, split_results in by_split_type.items():
            lines.append(f"### Split Type: `{split_type}`")
            lines.append("")

            for r in split_results:
                fold = r["fold"]
                fold_str = f"fold {fold}" if fold is not None else "default"
                lines.append(f"#### {split_type} — {fold_str}")
                lines.append("")
                lines.append(
                    f"- **Subjects:** train={r['subjects']['train']}, "
                    f"valid={r['subjects']['valid']}, "
                    f"test={r['subjects']['test']}"
                )
                lines.append(f"- **Recordings:** {r['n_recordings']}")
                lines.append("")

                # Remapped counts (main focus)
                lines.append("**Remapped Distribution:**")
                lines.append("")
                lines.append(make_counts_table(r, "remapped"))
                lines.append("")
                lines.append("**Imbalance:**")
                lines.append("")
                lines.append(make_imbalance_summary(r, "remapped"))
                lines.append("")

                # Proportion stability across splits
                lines.append(
                    "**Proportion Stability (train vs valid vs test):**"
                )
                lines.append("")
                lines.append(make_stability_table(r, "remapped"))
                lines.append("")

                # Train+Valid vs Test stability
                lines.append("**Train+Valid vs Test Stability:**")
                lines.append("")
                lines.append(make_trainval_test_stability_table(r, "remapped"))
                lines.append("")

            # Cross-fold comparison for multi-fold split types
            if len(split_results) > 1:
                lines.append(f"#### Cross-fold Comparison ({split_type})")
                lines.append("")
                lines.append(
                    "| Fold | Split | Total | Max/Min | CV | Largest | Smallest |"
                )
                lines.append("|---|---|---|---|---|---|---|")
                for r in split_results:
                    fold = r["fold"] if r["fold"] is not None else 0
                    for s in ["train", "valid", "test"]:
                        m = r["remapped"][s]["imbalance"]
                        total = m.get("total", 0)
                        ratio = m.get("max_min_ratio", 0)
                        cv = m.get("cv", 0)
                        max_cls = m.get("max_class", "")
                        max_cnt = m.get("max_count", 0)
                        min_cls = m.get("min_class", "")
                        min_cnt = m.get("min_count", 0)
                        lines.append(
                            f"| {fold} | {s} | {total} | "
                            f"{fmt_ratio(ratio)} | {cv:.3f} | "
                            f"{max_cls} ({max_cnt}) | "
                            f"{min_cls} ({min_cnt}) |"
                        )
                lines.append("")

                # Cross-fold imbalance summary
                lines.append("**Cross-fold Max/Min Ratio Range (remapped):**")
                lines.append("")
                lines.append("| Split | Min Ratio | Max Ratio | Mean Ratio |")
                lines.append("|---|---|---|---|")
                for s in ["train", "valid", "test"]:
                    ratios = [
                        r["remapped"][s]["imbalance"]["max_min_ratio"]
                        for r in split_results
                    ]
                    ratios_finite = [r for r in ratios if r != float("inf")]
                    if ratios_finite:
                        lines.append(
                            f"| {s} | {min(ratios_finite):.2f} | "
                            f"{max(ratios_finite):.2f} | "
                            f"{sum(ratios_finite) / len(ratios_finite):.2f} |"
                        )
                    else:
                        lines.append(f"| {s} | ∞ | ∞ | ∞ |")
                lines.append("")

    # Suggestions section
    lines.append("---")
    lines.append("")
    lines.append("## Analysis & Suggestions")
    lines.append("")

    # Auto-generate some analysis based on data
    for dataset_name, results in all_results.items():
        lines.append(f"### {dataset_name.upper()}")
        lines.append("")

        # Find worst imbalances
        worst_orig = {"split_type": "", "fold": 0, "split": "", "ratio": 0}
        worst_remap = {"split_type": "", "fold": 0, "split": "", "ratio": 0}
        best_remap = {
            "split_type": "",
            "fold": 0,
            "split": "",
            "ratio": float("inf"),
        }

        for r in results:
            for s in ["train", "valid", "test"]:
                orig_r = r["original"][s]["imbalance"]["max_min_ratio"]
                remap_r = r["remapped"][s]["imbalance"]["max_min_ratio"]
                if orig_r != float("inf") and orig_r > worst_orig["ratio"]:
                    worst_orig = {
                        "split_type": r["split_type"],
                        "fold": r["fold"],
                        "split": s,
                        "ratio": orig_r,
                    }
                if remap_r != float("inf") and remap_r > worst_remap["ratio"]:
                    worst_remap = {
                        "split_type": r["split_type"],
                        "fold": r["fold"],
                        "split": s,
                        "ratio": remap_r,
                    }
                if remap_r != float("inf") and remap_r < best_remap["ratio"]:
                    best_remap = {
                        "split_type": r["split_type"],
                        "fold": r["fold"],
                        "split": s,
                        "ratio": remap_r,
                    }

        lines.append(
            f"**Worst original imbalance:** {worst_orig['split_type']} fold "
            f"{worst_orig['fold']} {worst_orig['split']} "
            f"(max/min = {fmt_ratio(worst_orig['ratio'])})"
        )
        lines.append("")
        lines.append(
            f"**Worst remapped imbalance:** {worst_remap['split_type']} fold "
            f"{worst_remap['fold']} {worst_remap['split']} "
            f"(max/min = {fmt_ratio(worst_remap['ratio'])})"
        )
        lines.append("")
        lines.append(
            f"**Best remapped imbalance:** {best_remap['split_type']} fold "
            f"{best_remap['fold']} {best_remap['split']} "
            f"(max/min = {fmt_ratio(best_remap['ratio'])})"
        )
        lines.append("")

        # Analyze proportion drift across splits
        lines.append("**Proportion consistency across splits (remapped):**")
        lines.append("")
        max_drifts = []
        for r in results:
            stability = r["remapped"].get("trainval_test_stability", {})
            for label, info in stability.items():
                max_drifts.append({
                    "split_type": r["split_type"],
                    "fold": r["fold"],
                    "label": label,
                    "max_diff": info["max_diff"],
                })
        if max_drifts:
            max_drifts.sort(key=lambda x: x["max_diff"], reverse=True)
            lines.append(
                "Top 5 largest proportion drifts between train+valid and test:"
            )
            lines.append("")
            lines.append("| Split Type | Fold | Label | Max Diff (pp) |")
            lines.append("|---|---|---|---|")
            for d in max_drifts[:5]:
                lines.append(
                    f"| {d['split_type']} | {d['fold']} | "
                    f"{d['label']} | {d['max_diff']:.2f} |"
                )
            lines.append("")

    # General suggestions
    lines.append("---")
    lines.append("")
    lines.append("## General Recommendations")
    lines.append("")
    lines.append("### On the remapping itself")
    lines.append("")
    lines.append(
        "1. **The remapping significantly reduces class imbalance.** Going from 20+ classes"
    )
    lines.append(
        "   to 6 bands aggregates sparse frequencies, substantially improving the max/min ratio."
    )
    lines.append("")
    lines.append(
        "2. **Band sizes are unequal by design** (sub_bass=5 freqs, low=4, low_mid=3, mid=4,"
    )
    lines.append(
        "   high_mid=5, high=4). Since trial counts per frequency are roughly uniform within a"
    )
    lines.append(
        "   dataset, bands with more member frequencies will naturally have more samples."
    )
    lines.append(
        "   Consider whether this mapping reflects the perceptual/cochlear grouping you want"
    )
    lines.append(
        "   or if small adjustments (e.g., moving stim_1500Hz to low_mid) would yield better balance."
    )
    lines.append("")
    lines.append(
        "3. **White noise (`stim_wn`) is excluded** from the band mapping. If you want to"
    )
    lines.append(
        "   include it, it could be its own 7th class or dropped entirely."
    )
    lines.append("")
    lines.append("### On split balance")
    lines.append("")
    lines.append(
        "4. **Intersubject splits** show the most variability because different subjects"
    )
    lines.append(
        "   contribute different numbers of sessions/trials. The proportions within each"
    )
    lines.append(
        "   subject are usually consistent (same protocol), but total trial counts per class"
    )
    lines.append("   vary significantly across folds.")
    lines.append("")
    lines.append(
        "5. **Intersession splits** can show proportion drift if the stimulation protocol"
    )
    lines.append(
        "   changed between early and later sessions for the test subjects."
    )
    lines.append("")
    lines.append(
        "6. **Intrasession splits** (both block and causal) tend to have the most stable"
    )
    lines.append(
        "   proportions since trials are split within each recording, preserving the"
    )
    lines.append("   within-session class distribution.")
    lines.append("")
    lines.append("### Potential actions to improve balance")
    lines.append("")
    lines.append(
        "7. **Weighted sampling / loss weighting:** For training, use inverse-frequency"
    )
    lines.append(
        "   weights to counteract remaining imbalance. This is simpler than restructuring splits."
    )
    lines.append("")
    lines.append(
        "8. **Adjust band boundaries:** If one band dominates, consider splitting it or"
    )
    lines.append("   moving a boundary frequency to the adjacent band.")
    lines.append("")
    lines.append(
        "9. **Subject reassignment for intersubject:** If a particular subject has"
    )
    lines.append(
        "   disproportionately many/few trials of a band, placing them in test vs. train"
    )
    lines.append(
        "   will affect balance. Review per-subject trial counts to choose optimal"
    )
    lines.append("   test subjects that provide representative distributions.")
    lines.append("")

    report = "\n".join(lines)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Report saved to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="reports/label_analysis")
    parser.add_argument(
        "--output", default="reports/label_analysis/class_balance_report.md"
    )
    args = parser.parse_args()

    all_results = load_results(args.input_dir)
    generate_report(all_results, args.output)


if __name__ == "__main__":
    main()
