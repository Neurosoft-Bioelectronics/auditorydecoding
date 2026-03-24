"""Notebook-faithful linear pipeline from linear_training.ipynb (fixed flags).

PCA component ablation: windows are built once from the full normalized PCA
signal; each run zeros selected component channels on the window tensor
``(samples, time, components)`` before STFT / flatten / STD.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from einops import rearrange
from sklearn.preprocessing import LabelEncoder
from temporaldata import Data

from auditorydecoding.windowing import extract_windows

_LOG = logging.getLogger(__name__)

# Defaults aligned with notebooks/linear_training.ipynb
ZSCORE_BEFORE_PCA = False
WHITENING = True
N_COMPONENTS: int | None = None
NORM_BY_CHANNEL = False
WINDOW_LENGTH = 0.5
STFT = False
FLATTEN = True
STD = False
BALANCE_CLASSES = False
BALANCE_SEED = 42
LOGREG_MAX_ITER = 10000


@dataclass(frozen=True)
class AblationRunResult:
    name: str
    ablated_indices: tuple[int, ...]
    n_components: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class PcaAblationPrepared:
    """PCA fit once; ``signal_normalized`` is post-PCA norm (full recording)."""

    data: Data
    signal_normalized: np.ndarray
    timestamps: np.ndarray
    n_components: int


@dataclass(frozen=True)
class PcaAblationWindows:
    """Train/valid window tensors ``(n_windows, n_time, n_components)`` (PCA space)."""

    X_train_w: np.ndarray
    y_train: np.ndarray
    X_valid_w: np.ndarray
    y_valid: np.ndarray


def load_recording(data_root: str | Path, recording_id: str) -> Data:
    path = Path(data_root) / f"{recording_id}.h5"
    _LOG.debug("Opening %s", path.resolve())
    with h5py.File(path) as f:
        return Data.from_hdf5(f, lazy=False)


def apply_window_pca_ablation(
    X_windows: np.ndarray, ablated_indices: list[int] | tuple[int, ...]
) -> np.ndarray:
    """Zero PCA component channels on window array ``(samples, time, components)``."""
    if X_windows.ndim != 3:
        raise ValueError(
            f"Expected 3D windows (samples, time, components); got "
            f"{X_windows.shape}"
        )
    n_comp = X_windows.shape[2]
    ablated_sorted = tuple(sorted(set(ablated_indices)))
    for j in ablated_sorted:
        if j < 0 or j >= n_comp:
            raise ValueError(
                f"Ablated index {j} out of range for n_components={n_comp}"
            )
    if not ablated_sorted:
        return X_windows
    out = X_windows.copy()
    out[:, :, list(ablated_sorted)] = 0.0
    return out


def _maybe_zscore_train(data_train: np.ndarray) -> np.ndarray:
    if not ZSCORE_BEFORE_PCA:
        return data_train
    return (data_train - data_train.mean(axis=0)) / data_train.std(axis=0)


def _post_pca_normalize(
    signal_transformed: np.ndarray, train_transformed: np.ndarray
) -> np.ndarray:
    if not NORM_BY_CHANNEL:
        return signal_transformed
    mean = train_transformed.mean(axis=0)
    std = train_transformed.std(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    return (signal_transformed - mean) / std_safe


def _extract_features(X: np.ndarray) -> np.ndarray:
    """Match notebook: STFT -> FLATTEN -> STD (defaults STFT/STD off)."""
    if STFT:
        X = np.abs(np.fft.rfft(X, axis=1))
    if FLATTEN:
        X = rearrange(X, "s t c -> s (t c)")
    if STD:
        X = X.std(axis=1)
    return X


def _maybe_balance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_train_enc: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not BALANCE_CLASSES:
        return X_train, y_train, y_train_enc
    _LOG.debug("Undersampling train to balance classes (seed=%s)", seed)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y_train_enc, return_counts=True)
    min_count = int(counts.min())
    balanced_idx = np.concatenate(
        [
            rng.choice(
                np.where(y_train_enc == c)[0],
                size=min_count,
                replace=False,
            )
            for c in classes
        ]
    )
    balanced_idx.sort()
    return (
        X_train[balanced_idx],
        y_train[balanced_idx],
        y_train_enc[balanced_idx],
    )


def prepare_pca_ablation_basis(data: Data) -> PcaAblationPrepared:
    """Fit PCA on causal train and return full-recording normalized PCA signal."""
    keep_channels = data.channels.type == "ecog"
    train_intervals = data.splits.on_vs_off_causal_train.coalesce()
    chunks = [
        data.slice(float(iv[0]), float(iv[1])).ecog.signal[:, keep_channels]
        for iv in train_intervals
    ]
    data_train = np.concatenate(chunks, axis=0)
    data_train = _maybe_zscore_train(data_train)
    _LOG.debug(
        "PCA fit matrix data_train shape %s (samples x raw channels)",
        data_train.shape,
    )

    pca = PCA(whiten=WHITENING, n_components=N_COMPONENTS).fit(data_train)
    n_components = int(pca.n_components_)
    ev = pca.explained_variance_ratio_
    _LOG.debug(
        "PCA fitted: n_components=%d | explained_variance_ratio "
        "(first 5): %s | cumulative first 5: %.4f",
        n_components,
        np.array2string(ev[:5], precision=4, separator=", "),
        float(ev[:5].sum()) if ev.size else 0.0,
    )

    signal = data.ecog.signal[:, keep_channels]
    timestamps = np.asarray(data.ecog.timestamps)
    signal_t = pca.transform(signal)
    train_t = pca.transform(data_train)
    signal_normalized = _post_pca_normalize(signal_t, train_t)
    _LOG.debug(
        "Full recording transformed: raw signal %s -> PCA+norm %s | %d ecog channels",
        signal.shape,
        signal_normalized.shape,
        int(keep_channels.sum()),
    )

    return PcaAblationPrepared(
        data=data,
        signal_normalized=signal_normalized,
        timestamps=timestamps,
        n_components=n_components,
    )


def extract_pca_ablation_windows(
    prepared: PcaAblationPrepared,
) -> PcaAblationWindows:
    """Slice causal train/valid once from the full normalized PCA signal."""
    data = prepared.data
    ts = prepared.timestamps
    signal = prepared.signal_normalized
    train_iv = data.splits.on_vs_off_causal_train
    valid_iv = data.splits.on_vs_off_causal_valid

    X_train_w, y_train = extract_windows(signal, ts, train_iv, WINDOW_LENGTH)
    X_valid_w, y_valid = extract_windows(signal, ts, valid_iv, WINDOW_LENGTH)
    if X_train_w.shape[2] != X_valid_w.shape[2]:
        raise ValueError(
            "Train/valid window component counts differ: "
            f"{X_train_w.shape[2]} vs {X_valid_w.shape[2]}"
        )
    if X_train_w.shape[2] != prepared.n_components:
        raise ValueError(
            "Window last dim != PCA n_components: "
            f"{X_train_w.shape[2]} vs {prepared.n_components}"
        )
    _LOG.debug(
        "Windows extracted once: train %s valid %s | window_length=%s",
        X_train_w.shape,
        X_valid_w.shape,
        WINDOW_LENGTH,
    )
    return PcaAblationWindows(
        X_train_w=X_train_w,
        y_train=y_train,
        X_valid_w=X_valid_w,
        y_valid=y_valid,
    )


def run_pca_ablation_from_windows(
    windows: PcaAblationWindows,
    *,
    run_name: str,
    ablated_indices: list[int],
    balance_seed: int = BALANCE_SEED,
) -> AblationRunResult:
    n_components = windows.X_train_w.shape[2]
    ablated_sorted = tuple(sorted(set(ablated_indices)))
    for j in ablated_sorted:
        if j < 0 or j >= n_components:
            raise ValueError(
                f"ablated index {j} invalid for n_components={n_components}"
            )

    X_train_w = apply_window_pca_ablation(windows.X_train_w, ablated_sorted)
    X_valid_w = apply_window_pca_ablation(windows.X_valid_w, ablated_sorted)
    y_train = windows.y_train
    y_valid = windows.y_valid
    _LOG.debug(
        "Window PCA ablation: zeroed %d component(s) %s (before flatten)",
        len(ablated_sorted),
        list(ablated_sorted) if ablated_sorted else "[]",
    )

    le = LabelEncoder().fit(y_train)
    y_train_enc = le.transform(y_train)
    y_valid_enc = le.transform(y_valid)

    X_train = _extract_features(X_train_w)
    X_valid = _extract_features(X_valid_w)
    _LOG.debug(
        "Features (STFT=%s FLATTEN=%s STD=%s): train %s valid %s",
        STFT,
        FLATTEN,
        STD,
        X_train.shape,
        X_valid.shape,
    )

    X_train, y_train, y_train_enc = _maybe_balance(
        X_train, y_train, y_train_enc, balance_seed
    )
    _LOG.debug(
        "Train after balance: X %s | n_samples=%d", X_train.shape, len(y_train)
    )

    clf = LogisticRegression(max_iter=LOGREG_MAX_ITER)
    _LOG.debug("Fitting LogisticRegression(max_iter=%s)", LOGREG_MAX_ITER)
    clf.fit(X_train, y_train_enc)

    y_valid_pred = clf.predict(X_valid)

    metrics: dict[str, float] = {
        "valid_accuracy": float(accuracy_score(y_valid_enc, y_valid_pred)),
        "valid_balanced_accuracy": float(
            balanced_accuracy_score(y_valid_enc, y_valid_pred)
        ),
        "valid_f1_macro": float(
            f1_score(
                y_valid_enc,
                y_valid_pred,
                average="macro",
                zero_division=0,
            )
        ),
    }

    return AblationRunResult(
        name=run_name,
        ablated_indices=ablated_sorted,
        n_components=n_components,
        metrics=metrics,
    )
