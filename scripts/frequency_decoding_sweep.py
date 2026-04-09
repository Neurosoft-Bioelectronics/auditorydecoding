#!/usr/bin/env python3
"""Sweep bandpass filter frequencies for frequency-decoding pipeline.

Each (lowcut, highcut, stft_mode) combination is dispatched as an independent
Ray remote task.  Results are written to CSV (and optionally JSONL).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import ray
import yaml

from auditorydecoding.experiments.frequency_decoding_pipeline import (
    ExperimentResult,
    FrequencyDecodingConfig,
    run_experiment,
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
    p = argparse.ArgumentParser(
        description="Sweep bandpass frequencies with Ray parallelism."
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config file.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("frequency_sweep_results.csv"),
    )
    p.add_argument("--output-jsonl", type=Path, default=None)
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    p.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help=(
            "Max CPUs for Ray. Defaults to SLURM_CPUS_PER_TASK "
            "or os.cpu_count()."
        ),
    )
    p.add_argument(
        "--ray-address",
        default=None,
        help="Ray cluster address (default: local).",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override data_root from the YAML config.",
    )
    return p.parse_args()


def _load_config(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping")
    return raw


def _resolve_recording_ids(raw: dict) -> list[str]:
    """Extract recording ID(s) from the data section.

    Accepts either ``recording_id: "..."`` (single) or
    ``recording_ids: [...]`` (list).
    """
    data = raw.get("data", {})
    ids = data.get("recording_ids", data.get("recording_id"))
    if ids is None:
        raise ValueError(
            "Config must specify data.recording_ids (list) "
            "or data.recording_id (string)"
        )
    if isinstance(ids, str):
        return [ids]
    if isinstance(ids, list) and all(isinstance(x, str) for x in ids):
        return ids
    raise ValueError("recording_ids must be a string or list of strings")


def _base_config_from_yaml(raw: dict) -> FrequencyDecodingConfig:
    """Build a FrequencyDecodingConfig from the non-sweep YAML sections.

    ``recording_ids`` / ``recording_id`` are handled separately by the
    sweep builder; we use a placeholder here.
    """
    kw: dict = {}
    for section in ("data", "preprocessing", "features", "training"):
        kw.update(raw.get(section, {}))
    kw.pop("recording_ids", None)
    kw.pop("recording_id", None)
    return FrequencyDecodingConfig(**kw)


def _resolve_freq_values(cfg: dict) -> np.ndarray:
    """Build a 1-D array of frequencies from a sweep sub-config.

    Accepts two formats:
      - ``{"values": [1, 4, 8, ...]}``  — explicit list
      - ``{"min": ..., "max": ..., "num": ...}`` — geomspace grid
    """
    if "values" in cfg:
        return np.asarray(cfg["values"], dtype=float)
    return np.geomspace(cfg["min"], cfg["max"], cfg["num"])


def _generate_freq_grid(
    sweep: dict,
) -> list[tuple[float | None, float | None]]:
    """Return (lowcut, highcut) pairs including optional no-bandpass baseline."""
    pairs: list[tuple[float | None, float | None]] = []

    if sweep.get("include_no_bandpass", False):
        pairs.append((None, None))

    lowcuts = _resolve_freq_values(sweep["lowcut"])
    highcuts = _resolve_freq_values(sweep["highcut"])
    ratio = sweep.get("min_bandwidth_ratio", 2.0)

    for lo, hi in product(lowcuts, highcuts):
        if hi >= ratio * lo:
            pairs.append((float(lo), float(hi)))

    return pairs


def _build_run_configs(
    base: FrequencyDecodingConfig,
    sweep: dict,
    recording_ids: list[str],
) -> list[FrequencyDecodingConfig]:
    pairs = _generate_freq_grid(sweep)
    stft_modes: list[bool] = sweep.get("stft_modes", [True, False])

    configs: list[FrequencyDecodingConfig] = []
    for rec_id, (lo, hi), stft in product(recording_ids, pairs, stft_modes):
        configs.append(
            replace(
                base,
                recording_id=rec_id,
                bandpass_lowcut=lo,
                bandpass_highcut=hi,
                stft=stft,
            )
        )
    return configs


def _run_name(cfg: FrequencyDecodingConfig) -> str:
    lo = f"{cfg.bandpass_lowcut:.4g}" if cfg.bandpass_lowcut else "none"
    hi = f"{cfg.bandpass_highcut:.4g}" if cfg.bandpass_highcut else "none"
    stft_tag = "stft" if cfg.stft else "raw"
    return f"lo{lo}_hi{hi}_{stft_tag}"


def _resolve_num_cpus(num_cpus: int | None) -> int:
    if num_cpus is not None:
        return num_cpus
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm is not None:
        return int(slurm)
    return os.cpu_count() or 1


@ray.remote
def _ray_run_experiment(
    cfg: FrequencyDecodingConfig,
) -> ExperimentResult:
    return run_experiment(cfg)


_CSV_FIELDS = [
    "run_name",
    "recording_id",
    "lowcut",
    "highcut",
    "stft",
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "cohen_kappa",
    "bass",
    "n_classes",
    "n_valid_samples",
    "elapsed_seconds",
]

_INT_METRICS = {"n_classes", "n_valid_samples"}


def _write_results(
    results: list[ExperimentResult],
    csv_path: Path,
    jsonl_path: Path | None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in results:
            formatted_metrics = {
                k: (str(int(v)) if k in _INT_METRICS else f"{v:.6f}")
                for k, v in r.metrics.items()
            }
            w.writerow(
                {
                    "run_name": _run_name(r.config),
                    "recording_id": r.config.recording_id,
                    "lowcut": r.config.bandpass_lowcut or "",
                    "highcut": r.config.bandpass_highcut or "",
                    "stft": r.config.stft,
                    **formatted_metrics,
                    "elapsed_seconds": f"{r.elapsed_seconds:.2f}",
                }
            )
    _LOG.info("Wrote CSV: %s (%d rows)", csv_path.resolve(), len(results))

    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w") as f:
            for r in results:
                rec = {
                    "config": r.config.to_dict(),
                    "metrics": r.metrics,
                    "elapsed_seconds": r.elapsed_seconds,
                }
                f.write(json.dumps(rec) + "\n")
        _LOG.info("Wrote JSONL: %s", jsonl_path.resolve())


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)

    raw = _load_config(args.config)
    base = _base_config_from_yaml(raw)
    if args.data_root is not None:
        base = replace(base, data_root=str(args.data_root))
    recording_ids = _resolve_recording_ids(raw)
    sweep = raw.get("sweep", {})
    configs = _build_run_configs(base, sweep, recording_ids)
    num_cpus = _resolve_num_cpus(args.num_cpus)
    _LOG.info(
        "Generated %d experiment configs (%d recordings x sweep grid).",
        len(configs),
        len(recording_ids),
    )

    ray_kwargs: dict = {"num_cpus": num_cpus}
    if args.ray_address:
        ray_kwargs["address"] = args.ray_address
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if slurm_tmpdir:
        ray_kwargs["_temp_dir"] = slurm_tmpdir

    ray.init(**ray_kwargs)
    _LOG.info("Ray initialized: %s", ray.cluster_resources())

    t0 = time.monotonic()
    futures = [_ray_run_experiment.remote(cfg) for cfg in configs]

    results: list[ExperimentResult] = []
    remaining = list(futures)
    while remaining:
        done, remaining = ray.wait(remaining, num_returns=1)
        r = ray.get(done[0])
        results.append(r)
        _LOG.info(
            "[%d/%d] %s  rec=%s  acc=%.4f  bal_acc=%.4f  "
            "kappa=%.4f  bass=%.4f  (%.1fs)",
            len(results),
            len(futures),
            _run_name(r.config),
            r.config.recording_id,
            r.metrics["accuracy"],
            r.metrics["balanced_accuracy"],
            r.metrics["cohen_kappa"],
            r.metrics["bass"],
            r.elapsed_seconds,
        )

    wall = time.monotonic() - t0
    _LOG.info("All %d runs finished in %.1fs wall-clock.", len(results), wall)

    _write_results(results, args.output_csv, args.output_jsonl)
    ray.shutdown()


if __name__ == "__main__":
    main()
