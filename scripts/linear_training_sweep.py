from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from auditorydecoding.datasets.neurosoft_minipigs_2026.NeurosoftMinipgs2026 import (
    NeurosoftMinipigs2026,
)
from auditorydecoding.features import (
    FFTFeatures,
    FeatureExtractor,
    FlattenFeatures,
    MeanFeatures,
    StdFeatures,
)
from auditorydecoding.preprocessing import (
    ChannelSelector,
    PreprocessingPipeline,
    Resample,
    Whiten,
    ZeroCenter,
)
from auditorydecoding.windowing import extract_windows

FEATURE_EXTRACTOR_FACTORIES = {
    "flatten": FlattenFeatures,
    "fft": FFTFeatures,
    "mean": MeanFeatures,
    "std": StdFeatures,
}

OPTIONAL_PREPROCESSING_STEPS = ("zero_center", "whiten", "resample")
BALANCE_STRATEGIES = (
    "none",
    "undersample",
    "class_weight",
    "undersample+class_weight",
)


@dataclass(frozen=True)
class ExperimentConfig:
    pipeline_steps: tuple[str, ...]
    whiten_components: int | None
    balance_strategy: str

    @property
    def key(self) -> str:
        components = (
            "none"
            if self.whiten_components is None
            else str(self.whiten_components)
        )
        steps = ",".join(self.pipeline_steps) if self.pipeline_steps else "none"
        return (
            f"steps={steps}|"
            f"whiten_components={components}|"
            f"balance={self.balance_strategy}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run linear training sweeps over preprocessing, whitening, and "
            "class balancing settings."
        )
    )
    parser.add_argument(
        "--data-root",
        default="data/processed",
        help="Path passed as NeurosoftMinipigs2026(root=...).",
    )
    parser.add_argument(
        "--recording-id",
        default="sub-02_ses-01_task-AcousStim_acq-LH_desc-filtered",
        help="Single recording ID used by the notebook setup.",
    )
    parser.add_argument(
        "--window-length",
        type=float,
        default=0.5,
        help="Window length in seconds.",
    )
    parser.add_argument(
        "--fold-num",
        type=int,
        default=0,
        help="Fold number for dataset split lookup.",
    )
    parser.add_argument(
        "--split-type",
        default="intrasession",
        choices=["intrasession", "intersession", "intersubject"],
    )
    parser.add_argument(
        "--task-type",
        default="on_vs_off",
        choices=["on_vs_off", "acoustic_stim"],
    )
    parser.add_argument(
        "--feature-extractor",
        default="flatten",
        choices=tuple(FEATURE_EXTRACTOR_FACTORIES),
        help="Feature extractor used before the linear classifier.",
    )
    parser.add_argument(
        "--pipeline-grid",
        default="zero_center,whiten",
        help=(
            "Comma-separated optional step names to power-set over. "
            f"Allowed: {', '.join(OPTIONAL_PREPROCESSING_STEPS)}."
        ),
    )
    parser.add_argument(
        "--pipeline-spec",
        action="append",
        default=[],
        help=(
            "Explicit pipeline optional steps (comma-separated). "
            "Can be passed multiple times. Overrides --pipeline-grid."
        ),
    )
    parser.add_argument(
        "--whiten-components",
        nargs="+",
        default=["10"],
        help=(
            "Values to sweep when pipeline includes whiten. "
            "Use 'none' to keep all PCA components."
        ),
    )
    parser.add_argument(
        "--resample-rate",
        type=float,
        default=None,
        help="Target Hz for the resample step (required if resample is used).",
    )
    parser.add_argument(
        "--balance-strategies",
        nargs="+",
        default=["none", "undersample", "class_weight"],
        help=(
            "Class balancing strategies to sweep. "
            f"Allowed: {', '.join(BALANCE_STRATEGIES)}."
        ),
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="max_iter for LogisticRegression.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10_000,
        help="Number of permutations for balanced-accuracy null distribution.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for balancing and permutation tests.",
    )
    parser.add_argument(
        "--output-csv",
        default="linear_training_sweep_results.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="linear_training_sweep_results.jsonl",
        help="JSONL output path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top rows to print.",
    )
    parser.add_argument(
        "--rank-metric",
        default="test_combined_f1",
        help="Metric key used to rank and print top experiments.",
    )
    return parser.parse_args()


def parse_optional_steps(spec: str) -> tuple[str, ...]:
    if not spec.strip():
        return ()

    values = tuple(x.strip() for x in spec.split(",") if x.strip())
    unknown = set(values) - set(OPTIONAL_PREPROCESSING_STEPS)
    if unknown:
        allowed = ", ".join(OPTIONAL_PREPROCESSING_STEPS)
        raise ValueError(
            f"Unknown step(s): {sorted(unknown)}. Allowed: {allowed}"
        )
    return values


def parse_whiten_values(raw_values: list[str]) -> list[int | None]:
    parsed: list[int | None] = []
    for value in raw_values:
        normalized = value.strip().lower()
        if normalized == "none":
            parsed.append(None)
            continue
        parsed.append(int(normalized))
    return parsed


def parse_balance_strategies(values: list[str]) -> list[str]:
    unknown = set(values) - set(BALANCE_STRATEGIES)
    if unknown:
        allowed = ", ".join(BALANCE_STRATEGIES)
        raise ValueError(
            f"Unknown balance strategy(ies): {sorted(unknown)}. Allowed: {allowed}"
        )
    return values


def generate_pipeline_options(
    args: argparse.Namespace,
) -> list[tuple[str, ...]]:
    if args.pipeline_spec:
        options = [parse_optional_steps(spec) for spec in args.pipeline_spec]
        return sorted(set(options))

    candidates = parse_optional_steps(args.pipeline_grid)
    options = []
    for use_mask in itertools.product([False, True], repeat=len(candidates)):
        selected = tuple(
            step for step, use_step in zip(candidates, use_mask) if use_step
        )
        options.append(selected)
    return sorted(set(options))


def build_pipeline(
    optional_steps: tuple[str, ...],
    whiten_components: int | None,
    resample_rate: float | None,
) -> PreprocessingPipeline:
    steps = [ChannelSelector(channel_types=["ecog"])]
    for name in optional_steps:
        if name == "zero_center":
            steps.append(ZeroCenter())
        elif name == "whiten":
            steps.append(Whiten(n_components=whiten_components))
        elif name == "resample":
            if resample_rate is None:
                raise ValueError(
                    "Pipeline contains 'resample' but --resample-rate is not set."
                )
            steps.append(Resample(target_rate=resample_rate))
        else:
            raise ValueError(f"Unexpected preprocessing step: {name}")
    return PreprocessingPipeline(steps)


def get_feature_extractor(name: str) -> FeatureExtractor:
    return FEATURE_EXTRACTOR_FACTORIES[name]()


def maybe_balance_train_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_train_enc: np.ndarray,
    strategy: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "undersample" not in strategy:
        return X_train, y_train, y_train_enc

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y_train_enc, return_counts=True)
    min_count = counts.min()
    selected_idx = np.concatenate(
        [
            rng.choice(
                np.where(y_train_enc == class_id)[0],
                size=min_count,
                replace=False,
            )
            for class_id in classes
        ]
    )
    selected_idx.sort()
    return (
        X_train[selected_idx],
        y_train[selected_idx],
        y_train_enc[selected_idx],
    )


def run_experiment(
    args: argparse.Namespace,
    config: ExperimentConfig,
) -> dict[str, Any]:
    preprocessing = build_pipeline(
        optional_steps=config.pipeline_steps,
        whiten_components=config.whiten_components,
        resample_rate=args.resample_rate,
    )
    dataset = NeurosoftMinipigs2026(
        root=args.data_root,
        recording_ids=[args.recording_id],
        fold_num=args.fold_num,
        split_type=args.split_type,
        task_type=args.task_type,
        preprocessing=preprocessing,
    )
    dataset.preprocess(split="train", plot=False)

    feature_extractor = get_feature_extractor(args.feature_extractor)
    X_train, y_train = extract_windows(
        dataset,
        "train",
        args.window_length,
        feature_extractor=feature_extractor,
    )
    X_valid, y_valid = extract_windows(
        dataset,
        "valid",
        args.window_length,
        feature_extractor=feature_extractor,
    )
    X_test, y_test = extract_windows(
        dataset,
        "test",
        args.window_length,
        feature_extractor=feature_extractor,
    )

    label_encoder = LabelEncoder().fit(y_train)
    y_train_enc = label_encoder.transform(y_train)
    y_valid_enc = label_encoder.transform(y_valid)
    y_test_enc = label_encoder.transform(y_test)

    X_train, y_train, y_train_enc = maybe_balance_train_set(
        X_train=X_train,
        y_train=y_train,
        y_train_enc=y_train_enc,
        strategy=config.balance_strategy,
        seed=args.seed,
    )

    class_weight = (
        "balanced" if "class_weight" in config.balance_strategy else None
    )
    classifier = LogisticRegression(
        max_iter=args.max_iter, class_weight=class_weight
    )
    classifier.fit(X_train, y_train_enc)

    y_valid_pred = classifier.predict(X_valid)
    y_test_pred = classifier.predict(X_test)

    val_acc = accuracy_score(y_valid_enc, y_valid_pred)
    test_acc = accuracy_score(y_test_enc, y_test_pred)
    val_bal_acc = balanced_accuracy_score(y_valid_enc, y_valid_pred)
    test_bal_acc = balanced_accuracy_score(y_test_enc, y_test_pred)
    val_precision_macro = precision_score(
        y_valid_enc,
        y_valid_pred,
        average="macro",
        zero_division=0,
    )
    test_precision_macro = precision_score(
        y_test_enc,
        y_test_pred,
        average="macro",
        zero_division=0,
    )
    val_recall_macro = recall_score(
        y_valid_enc,
        y_valid_pred,
        average="macro",
        zero_division=0,
    )
    test_recall_macro = recall_score(
        y_test_enc,
        y_test_pred,
        average="macro",
        zero_division=0,
    )
    val_macro_f1 = f1_score(y_valid_enc, y_valid_pred, average="macro")
    test_macro_f1 = f1_score(y_test_enc, y_test_pred, average="macro")

    combined_f1 = test_macro_f1
    n_correct = int((y_test_pred == y_test_enc).sum())
    n_total = int(y_test_enc.size)
    class_counts = np.bincount(y_test_enc)
    majority_chance = class_counts.max() / class_counts.sum()

    binom_pvalue = binomtest(
        n_correct,
        n_total,
        p=majority_chance,
        alternative="greater",
    ).pvalue

    rng = np.random.default_rng(args.seed)
    null_bal_accs = np.array(
        [
            balanced_accuracy_score(rng.permutation(y_test_enc), y_test_pred)
            for _ in range(args.n_permutations)
        ]
    )
    permutation_pvalue = float((null_bal_accs >= test_bal_acc).mean())

    result: dict[str, Any] = {
        "experiment_key": config.key,
        "pipeline_steps": ",".join(config.pipeline_steps)
        if config.pipeline_steps
        else "none",
        "whiten_components": config.whiten_components,
        "balance_strategy": config.balance_strategy,
        "feature_extractor": args.feature_extractor,
        "class_weight": class_weight or "none",
        "n_train": int(X_train.shape[0]),
        "n_valid": int(X_valid.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "val_balanced_accuracy": float(val_bal_acc),
        "test_balanced_accuracy": float(test_bal_acc),
        "val_precision_macro": float(val_precision_macro),
        "test_precision_macro": float(test_precision_macro),
        "val_recall_macro": float(val_recall_macro),
        "test_recall_macro": float(test_recall_macro),
        "val_macro_f1": float(val_macro_f1),
        "test_macro_f1": float(test_macro_f1),
        "test_combined_f1": float(combined_f1),
        "majority_chance_accuracy": float(majority_chance),
        "binomial_pvalue": float(binom_pvalue),
        "permutation_pvalue_balanced_accuracy": permutation_pvalue,
    }

    if len(label_encoder.classes_) == 2:
        for class_index, class_name in enumerate(label_encoder.classes_):
            class_f1 = f1_score(
                y_test_enc == class_index,
                y_test_pred == class_index,
                average="binary",
                zero_division=0,
            )
            result[f"test_f1_{class_name}"] = float(class_f1)

    return result


def save_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_results_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    whiten_values = parse_whiten_values(args.whiten_components)
    balance_strategies = parse_balance_strategies(args.balance_strategies)
    pipeline_options = generate_pipeline_options(args)

    configs: list[ExperimentConfig] = []
    for pipeline_steps in pipeline_options:
        if "resample" in pipeline_steps and args.resample_rate is None:
            raise ValueError(
                "At least one pipeline contains 'resample' but --resample-rate "
                "is not set."
            )

        if "whiten" in pipeline_steps:
            components_to_try = whiten_values
        else:
            components_to_try = [None]

        for components in components_to_try:
            for strategy in balance_strategies:
                configs.append(
                    ExperimentConfig(
                        pipeline_steps=pipeline_steps,
                        whiten_components=components,
                        balance_strategy=strategy,
                    )
                )

    print(f"Prepared {len(configs)} experiment(s).")
    results: list[dict[str, Any]] = []
    for index, config in enumerate(configs, start=1):
        print(f"[{index:03d}/{len(configs):03d}] {config.key}")
        row = run_experiment(args=args, config=config)
        results.append(row)
        print(
            "      "
            f"test_acc={row['test_accuracy']:.3f} "
            f"test_bal_acc={row['test_balanced_accuracy']:.3f} "
            f"test_combined_f1={row['test_combined_f1']:.3f}"
        )

    if not results:
        print("No experiments were generated.")
        return

    output_csv = Path(args.output_csv)
    output_jsonl = Path(args.output_jsonl)
    save_results_csv(output_csv, results)
    save_results_jsonl(output_jsonl, results)

    rank_metric = args.rank_metric
    ranked = sorted(
        results,
        key=lambda row: row.get(rank_metric, float("-inf")),
        reverse=True,
    )

    print()
    print(f"Top {min(args.top_k, len(ranked))} by '{rank_metric}':")
    for row in ranked[: args.top_k]:
        print(
            f"- {row['experiment_key']} | "
            f"{rank_metric}={row.get(rank_metric)} | "
            f"test_acc={row['test_accuracy']:.4f} | "
            f"test_bal_acc={row['test_balanced_accuracy']:.4f}"
        )

    print()
    print(f"Wrote CSV:   {output_csv.resolve()}")
    print(f"Wrote JSONL: {output_jsonl.resolve()}")


if __name__ == "__main__":
    main()
