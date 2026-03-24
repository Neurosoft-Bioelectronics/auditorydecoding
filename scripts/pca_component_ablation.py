#!/usr/bin/env python3
"""PCA component ablation study (notebook-faithful pipeline, variable mask only)."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import yaml

from auditorydecoding.experiments import notebook_linear_pipeline as nlp
from auditorydecoding.experiments.notebook_linear_pipeline import (
    extract_pca_ablation_windows,
    load_recording,
    prepare_pca_ablation_basis,
    run_pca_ablation_from_windows,
)

_LOG = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PCA component ablations: same pipeline as "
            "notebooks/linear_training.ipynb; only ablated PCA indices differ."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--yaml",
        type=Path,
        help="YAML file with a top-level 'runs' list (name + ablate).",
    )
    src.add_argument(
        "--preset",
        choices=("leave-one-out", "pairs"),
        help=(
            "leave-one-out: baseline + one run per component index. "
            "pairs: all unordered pairs (use --allow-large to confirm)."
        ),
    )
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help="Required for --preset pairs (many runs).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/processed/neurosoft_minipigs_2026"),
        help="Directory containing {recording_id}.h5",
    )
    parser.add_argument(
        "--recording-id",
        default="sub-02_ses-01_task-AcousStim_acq-LH_desc-filtered",
        help="Recording id (filename without .h5).",
    )
    parser.add_argument(
        "--balance-seed",
        type=int,
        default=42,
        help="RNG seed for optional class balancing (notebook default 42).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("pca_component_ablation_results.csv"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL path (one object per run).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default INFO).",
    )
    return parser.parse_args()


def _load_runs_yaml(path: Path) -> list[tuple[str, list[int]]]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "runs" not in raw:
        raise ValueError("YAML must be a mapping with a 'runs' key")
    runs_raw = raw["runs"]
    if not isinstance(runs_raw, list) or not runs_raw:
        raise ValueError("'runs' must be a non-empty list")
    out: list[tuple[str, list[int]]] = []
    for i, item in enumerate(runs_raw):
        if not isinstance(item, dict):
            raise ValueError(f"runs[{i}] must be a mapping")
        name = item.get("name")
        ablate = item.get("ablate")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"runs[{i}].name must be a non-empty string")
        if ablate is None:
            ablate = []
        if not isinstance(ablate, list) or not all(
            isinstance(x, int) for x in ablate
        ):
            raise ValueError(
                f"runs[{i}].ablate must be a list of integers (or empty)"
            )
        out.append((name, list(ablate)))
    return out


def _preset_runs(
    preset: str, n_components: int, allow_large: bool
) -> list[tuple[str, list[int]]]:
    if preset == "leave-one-out":
        runs = [("baseline_all_components", [])]
        runs.extend((f"drop_pc_{j}", [j]) for j in range(n_components))
        return runs
    if preset == "pairs":
        n_pairs = n_components * (n_components - 1) // 2
        if not allow_large:
            logging.error(
                "Refusing %d pair runs for --preset pairs "
                "(n_components=%d). Pass --allow-large.",
                n_pairs,
                n_components,
            )
            sys.exit(1)
        runs = [("baseline_all_components", [])]
        for i in range(n_components):
            for j in range(i + 1, n_components):
                runs.append((f"drop_pc_{i}_{j}", [i, j]))
        return runs
    raise ValueError(f"Unknown preset {preset!r}")


def _validate_ablate(indices: list[int], n_components: int) -> None:
    for j in indices:
        if j < 0 or j >= n_components:
            raise ValueError(
                f"ablated index {j} out of range [0, {n_components})"
            )


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)

    h5_path = args.data_root / f"{args.recording_id}.h5"
    _LOG.info(
        "PCA component ablation | recording=%s | data_root=%s | h5=%s",
        args.recording_id,
        args.data_root.resolve(),
        h5_path.resolve(),
    )

    _LOG.info("Loading HDF5…")
    data = load_recording(args.data_root, args.recording_id)
    _LOG.info("Loaded Data object.")

    _LOG.info(
        "Fitting PCA once on causal train (notebook defaults: "
        "zscore_before_pca=%s, whiten=%s, norm_after_pca=%s, window=%ss)…",
        nlp.ZSCORE_BEFORE_PCA,
        nlp.WHITENING,
        nlp.NORM_BY_CHANNEL,
        nlp.WINDOW_LENGTH,
    )
    prepared = prepare_pca_ablation_basis(data)
    n_comp = prepared.n_components
    sig = prepared.signal_normalized
    _LOG.info(
        "PCA basis ready: n_components=%d | normalized signal shape %s "
        "(timepoints x components) | %d timestamps",
        n_comp,
        sig.shape,
        prepared.timestamps.size,
    )

    _LOG.info(
        "Extracting train/valid windows once (ablations zero PCA channels on "
        "(samples, time, components) before flattening)…"
    )
    windows = extract_pca_ablation_windows(prepared)
    _LOG.info(
        "Windows cached: train %s valid %s",
        windows.X_train_w.shape,
        windows.X_valid_w.shape,
    )

    if args.yaml is not None:
        runs = _load_runs_yaml(args.yaml)
        _LOG.info(
            "Loaded %d run(s) from YAML %s", len(runs), args.yaml.resolve()
        )
    else:
        runs = _preset_runs(args.preset, n_comp, args.allow_large)
        _LOG.info(
            "Using preset %r: %d run(s) (n_components=%d)",
            args.preset,
            len(runs),
            n_comp,
        )

    results = []
    for run_idx, (name, ablate) in enumerate(runs, start=1):
        _validate_ablate(ablate, n_comp)
        _LOG.info(
            "[%d/%d] %s | ablate=%s",
            run_idx,
            len(runs),
            name,
            ablate if ablate else "(none — full PCA)",
        )
        result = run_pca_ablation_from_windows(
            windows,
            run_name=name,
            ablated_indices=ablate,
            balance_seed=args.balance_seed,
        )
        results.append(result)
        m = result.metrics
        _LOG.info(
            "    valid: accuracy=%.4f balanced_acc=%.4f f1_macro=%.4f",
            m["valid_accuracy"],
            m["valid_balanced_accuracy"],
            m["valid_f1_macro"],
        )

    baseline_f1_valid: float | None = None
    for r in results:
        if r.ablated_indices == ():
            baseline_f1_valid = r.metrics["valid_f1_macro"]
            break
    if baseline_f1_valid is not None:
        _LOG.info(
            "Baseline F1 (macro, valid) for deltas: %.4f (first run with ablate=[])",
            baseline_f1_valid,
        )
    else:
        _LOG.warning(
            "No run with ablate=[]; delta_valid_f1_macro columns will be empty."
        )

    fieldnames = [
        "name",
        "ablated_components",
        "n_components",
        "valid_accuracy",
        "valid_balanced_accuracy",
        "valid_f1_macro",
        "delta_valid_f1_macro",
    ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {
                "name": r.name,
                "ablated_components": json.dumps(list(r.ablated_indices)),
                "n_components": r.n_components,
                **{k: f"{r.metrics[k]:.6f}" for k in r.metrics},
            }
            if baseline_f1_valid is not None:
                row["delta_valid_f1_macro"] = (
                    f"{r.metrics['valid_f1_macro'] - baseline_f1_valid:.6f}"
                )
            else:
                row["delta_valid_f1_macro"] = ""
            w.writerow(row)
    _LOG.info(
        "Wrote CSV: %s (%d rows)", args.output_csv.resolve(), len(results)
    )

    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w") as fj:
            for r in results:
                rec = {
                    "name": r.name,
                    "ablated_indices": list(r.ablated_indices),
                    "n_components": r.n_components,
                    "metrics": r.metrics,
                }
                if baseline_f1_valid is not None:
                    rec["delta_valid_f1_macro"] = (
                        r.metrics["valid_f1_macro"] - baseline_f1_valid
                    )
                fj.write(json.dumps(rec) + "\n")
        _LOG.info("Wrote JSONL: %s", args.output_jsonl.resolve())

    _LOG.info("Done (%d ablation run(s)).", len(results))


if __name__ == "__main__":
    main()
