"""Parameterized frequency-decoding pipeline from frequency_decoding.ipynb.

Every tunable knob lives in :class:`FrequencyDecodingConfig`; no module-level
mutable state.  ``run_experiment`` is the single entry-point that runs the
full load -> filter -> PCA -> window -> features -> LogReg -> evaluate chain.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import h5py
import numpy as np
from einops import rearrange
from scipy.signal import butter, decimate as sp_decimate, sosfilt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from temporaldata import Data

from auditorydecoding.windowing import extract_windows

_LOG = logging.getLogger(__name__)

_SPLIT_KEYS = {
    "acoustic_stim": (
        "acoustic_stim_causal_train",
        "acoustic_stim_causal_valid",
    ),
    "on_vs_off": (
        "on_vs_off_causal_train",
        "on_vs_off_causal_valid",
    ),
}


@dataclass(frozen=True)
class FrequencyDecodingConfig:
    data_root: str = (
        "/network/scratch/s/sobralm/brainsets/processed/neurosoft_minipigs_2026"
    )
    recording_id: str = "sub-02_ses-01_task-AcousStim_acq-LH_desc-raw"
    task_type: str = "acoustic_stim"
    window_length: float = 0.5

    bandpass_lowcut: float | None = 0.1
    bandpass_highcut: float | None = 50.0
    bandpass_fs: int = 2000
    bandpass_order: int = 6

    zscore_before_pca: bool = False
    whitening: bool = False
    apply_pca: bool = True
    n_components: int | None = None
    norm_by_channel: bool = False

    decimate_factor: int = 1

    stft: bool = False
    stft_log: bool = True
    crop_freqs: bool = True

    zscore_features: bool = False
    std_normalize: bool = True
    flatten: bool = True

    balance_classes: bool = False
    balance_seed: int = 42
    logreg_max_iter: int = 10_000

    n_permutations: int = 10_000
    permutation_seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentResult:
    config: FrequencyDecodingConfig
    metrics: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def bandpass_filter(
    signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: int,
    order: int = 6,
) -> np.ndarray:
    nyq = 0.5 * fs
    sos = butter(
        order,
        [lowcut / nyq, highcut / nyq],
        btype="band",
        output="sos",
    )
    if signal.ndim == 1:
        return sosfilt(sos, signal)
    return np.column_stack(
        [sosfilt(sos, signal[:, ch]) for ch in range(signal.shape[1])]
    )


# ---------------------------------------------------------------------------
# Pipeline stages – each takes explicit arguments, no global state
# ---------------------------------------------------------------------------


def load_recording(data_root: str | Path, recording_id: str) -> Data:
    path = Path(data_root) / f"{recording_id}.h5"
    _LOG.debug("Opening %s", path.resolve())
    with h5py.File(path) as f:
        return Data.from_hdf5(f, lazy=False)


def _resolve_splits(data: Data, task_type: str):
    if task_type not in _SPLIT_KEYS:
        raise ValueError(
            f"Unknown task_type {task_type!r}; choose from {list(_SPLIT_KEYS)}"
        )
    train_key, valid_key = _SPLIT_KEYS[task_type]
    return getattr(data.splits, train_key), getattr(data.splits, valid_key)


def _apply_bandpass(
    signal: np.ndarray,
    cfg: FrequencyDecodingConfig,
) -> np.ndarray:
    if cfg.bandpass_lowcut is None or cfg.bandpass_highcut is None:
        return signal
    return bandpass_filter(
        signal,
        cfg.bandpass_lowcut,
        cfg.bandpass_highcut,
        cfg.bandpass_fs,
        cfg.bandpass_order,
    )


def _fit_pca_and_transform(
    signal: np.ndarray,
    data_train: np.ndarray,
    cfg: FrequencyDecodingConfig,
) -> np.ndarray:
    if not cfg.apply_pca:
        return signal

    if cfg.zscore_before_pca:
        mean, std = data_train.mean(axis=0), data_train.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        data_train = (data_train - mean) / std

    pca = PCA(whiten=cfg.whitening, n_components=cfg.n_components)
    pca.fit(data_train)
    _LOG.debug(
        "PCA: n_components=%d  explained_var (first 5): %s",
        pca.n_components_,
        np.array2string(
            pca.explained_variance_ratio_[:5], precision=4, separator=", "
        ),
    )

    signal_t = pca.transform(signal)

    if cfg.norm_by_channel:
        train_t = pca.transform(data_train)
        mean = train_t.mean(axis=0)
        std = train_t.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        signal_t = (signal_t - mean) / std

    return signal_t


def _extract_features(
    X: np.ndarray,
    cfg: FrequencyDecodingConfig,
) -> np.ndarray:
    """decimate -> STFT -> crop -> zscore -> std_normalize -> flatten (all optional)."""
    if cfg.decimate_factor > 1:
        X = sp_decimate(X, cfg.decimate_factor, axis=1)

    if cfg.stft:
        X = np.abs(np.fft.rfft(X, axis=1))
        if cfg.stft_log:
            X = np.log(X + 1e-12)
        if cfg.crop_freqs and cfg.bandpass_lowcut and cfg.bandpass_highcut:
            freqs = np.fft.rfftfreq(
                int(cfg.window_length * cfg.bandpass_fs),
                d=1 / cfg.bandpass_fs,
            )
            valid = (freqs >= cfg.bandpass_lowcut) & (
                freqs <= cfg.bandpass_highcut
            )
            X = X[:, valid, :]

    if cfg.zscore_features:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - mean) / std

    if cfg.std_normalize:
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-8
        X = (X - mean) / std

    if cfg.flatten:
        X = rearrange(X, "s t c -> s (c t)")

    return X


def _maybe_balance(
    X: np.ndarray,
    y: np.ndarray,
    y_enc: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y_enc, return_counts=True)
    min_count = int(counts.min())
    idx = np.concatenate(
        [
            rng.choice(np.where(y_enc == c)[0], size=min_count, replace=False)
            for c in classes
        ]
    )
    idx.sort()
    return X[idx], y[idx], y_enc[idx]


def _permutation_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    observed_bal_acc: float,
    n_permutations: int,
    seed: int,
) -> tuple[float, float]:
    """Permutation test on balanced accuracy.

    Returns (p_value, z_score) where z_score = (observed - null_mean) / null_std.
    """
    rng = np.random.default_rng(seed)
    null_accs = np.array(
        [
            float(balanced_accuracy_score(rng.permutation(y_true), y_pred))
            for _ in range(n_permutations)
        ]
    )
    p_value = float((null_accs >= observed_bal_acc).mean())
    null_std = float(null_accs.std())
    if null_std < 1e-12:
        z_score = 0.0
    else:
        z_score = float((observed_bal_acc - null_accs.mean()) / null_std)
    return p_value, z_score


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def run_experiment(cfg: FrequencyDecodingConfig) -> ExperimentResult:
    """Run the full frequency-decoding pipeline for a single config."""
    t0 = time.monotonic()

    data = load_recording(cfg.data_root, cfg.recording_id)
    keep = data.channels.type == "ecog"
    signal = data.ecog.signal[:, keep]
    timestamps = np.asarray(data.ecog.timestamps)

    train_iv, valid_iv = _resolve_splits(data, cfg.task_type)
    train_chunks = [
        data.slice(float(iv[0]), float(iv[1])).ecog.signal[:, keep]
        for iv in train_iv.coalesce()
    ]
    data_train_raw = np.concatenate(train_chunks, axis=0)

    signal = _apply_bandpass(signal, cfg)
    data_train_filtered = _apply_bandpass(data_train_raw, cfg)

    signal = _fit_pca_and_transform(signal, data_train_filtered, cfg)

    X_train_w, y_train = extract_windows(
        signal, timestamps, train_iv, cfg.window_length
    )
    X_valid_w, y_valid = extract_windows(
        signal, timestamps, valid_iv, cfg.window_length
    )
    _LOG.debug("Windows: train %s  valid %s", X_train_w.shape, X_valid_w.shape)

    le = LabelEncoder().fit(y_train)
    y_train_enc = le.transform(y_train)
    y_valid_enc = le.transform(y_valid)

    X_train = _extract_features(X_train_w, cfg)
    X_valid = _extract_features(X_valid_w, cfg)
    _LOG.debug("Features: train %s  valid %s", X_train.shape, X_valid.shape)

    if cfg.balance_classes:
        X_train, y_train, y_train_enc = _maybe_balance(
            X_train, y_train, y_train_enc, cfg.balance_seed
        )

    clf = LogisticRegression(max_iter=cfg.logreg_max_iter)
    clf.fit(X_train, y_train_enc)
    y_pred = clf.predict(X_valid)

    n_classes = len(le.classes_)
    bal_acc = float(balanced_accuracy_score(y_valid_enc, y_pred))

    perm_pvalue, perm_zscore = _permutation_test(
        y_valid_enc,
        y_pred,
        bal_acc,
        cfg.n_permutations,
        cfg.permutation_seed,
    )

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_valid_enc, y_pred)),
        "balanced_accuracy": bal_acc,
        "f1_macro": float(
            f1_score(y_valid_enc, y_pred, average="macro", zero_division=0)
        ),
        "cohen_kappa": float(cohen_kappa_score(y_valid_enc, y_pred)),
        "perm_pvalue": perm_pvalue,
        "perm_zscore": perm_zscore,
        "n_classes": n_classes,
        "n_valid_samples": len(y_valid_enc),
    }

    elapsed = time.monotonic() - t0
    _LOG.info(
        "Done [%.1fs]  lowcut=%s highcut=%s stft=%s  "
        "acc=%.4f  bal_acc=%.4f  kappa=%.4f  perm_z=%.2f  perm_p=%.4g",
        elapsed,
        cfg.bandpass_lowcut,
        cfg.bandpass_highcut,
        cfg.stft,
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["cohen_kappa"],
        perm_zscore,
        perm_pvalue,
    )
    return ExperimentResult(
        config=cfg, metrics=metrics, elapsed_seconds=elapsed
    )
