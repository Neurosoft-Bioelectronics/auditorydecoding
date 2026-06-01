from __future__ import annotations

import logging

import numpy as np

from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_windows(
    signal: np.ndarray,
    timestamps: np.ndarray,
    intervals,
    window_length: float,
    label_key: str = "behavior_labels",
) -> tuple[np.ndarray, np.ndarray]:
    """Slice *signal* into fixed-length windows defined by *intervals*.

    The function iterates over each entry in *intervals*, creates sequential
    non-overlapping windows of *window_length* seconds, and collects the
    corresponding signal and label for each window.

    If an interval is shorter than *window_length* it is skipped (with a
    warning).  After the last regular window that fits, a final window aligned
    to the interval end is added when there is a remainder -- this mirrors the
    ``SequentialFixedWindowSampler`` behaviour and ensures full interval
    coverage (at the cost of a possible overlap with the previous window).

    Parameters
    ----------
    signal
        Array of shape ``(T, C)`` — the (possibly preprocessed) signal.
        Preprocessing such as PCA or whitening can be applied beforehand;
        the number of timepoints must still match *timestamps*.
    timestamps
        Array of shape ``(T,)`` with monotonically increasing sample times
        (in seconds).
    intervals
        A ``temporaldata.Interval`` (or compatible object) with at least
        ``start``, ``end``, and an attribute named *label_key* (by default
        ``behavior_labels``).  The split-level interval objects stored on
        the ``Data`` object (e.g. ``data.splits.on_vs_off_causal_train``)
        already carry these fields.
    window_length
        Duration of each window in seconds.
    label_key
        Name of the attribute on *intervals* that carries per-interval
        labels.
    """
    labels_array = getattr(intervals, label_key)

    X: list[np.ndarray] = []
    y: list = []
    target_num_samples: int | None = None

    skipped_seconds = 0.0

    for i in tqdm(range(len(intervals)), desc="Extracting windows"):
        iv_start: float = float(intervals.start[i])
        iv_end: float = float(intervals.end[i])
        iv_duration = iv_end - iv_start

        if iv_duration < window_length:
            skipped_seconds += iv_duration
            continue

        label = labels_array[i]

        win_start = iv_start
        while win_start + window_length <= iv_end:
            win_end = win_start + window_length
            mask = (timestamps >= win_start) & (timestamps < win_end)
            window_signal = signal[mask]

            if target_num_samples is None:
                target_num_samples = int(window_signal.shape[0])

            window_signal = _ensure_window_length(
                window_signal,
                target_num_samples,
                win_start=win_start,
                win_end=win_end,
            )
            X.append(window_signal)
            y.append(label)

            win_start = win_end

        remainder = iv_end - win_start
        if remainder > 1e-9:
            tail_start = iv_end - window_length
            if tail_start >= iv_start and tail_start > (
                win_start - window_length + 1e-9
            ):
                mask = (timestamps >= tail_start) & (timestamps < iv_end)
                window_signal = signal[mask]
                if target_num_samples is not None:
                    window_signal = _ensure_window_length(
                        window_signal,
                        target_num_samples,
                        win_start=tail_start,
                        win_end=iv_end,
                    )
                X.append(window_signal)
                y.append(label)

    if skipped_seconds > 0:
        total = sum(
            float(intervals.end[i]) - float(intervals.start[i])
            for i in range(len(intervals))
        )
        remaining = total - skipped_seconds
        logger.warning(
            "Skipping %.4f seconds of data due to short intervals. "
            "Remaining: %.1f seconds.",
            skipped_seconds,
            remaining,
        )

    return np.stack(X), np.array(y)


def _ensure_window_length(
    signal: np.ndarray,
    target_num_samples: int,
    *,
    win_start: float,
    win_end: float,
) -> np.ndarray:
    """Resample *signal* to exactly *target_num_samples* timepoints."""
    if signal.ndim != 2:
        raise ValueError(
            "Expected 2D signal windows (time, channels), "
            f"got shape {signal.shape}."
        )

    n_timepoints, n_channels = signal.shape
    if n_timepoints == target_num_samples:
        return signal

    if n_timepoints == 0:
        raise ValueError(
            "Encountered an empty window while extracting windows. "
            f"start={win_start}, end={win_end}"
        )

    old_axis = np.linspace(0.0, 1.0, n_timepoints, dtype=np.float64)
    new_axis = np.linspace(0.0, 1.0, target_num_samples, dtype=np.float64)
    resized = np.empty((target_num_samples, n_channels), dtype=signal.dtype)
    for channel_idx in range(n_channels):
        resized[:, channel_idx] = np.interp(
            new_axis, old_axis, signal[:, channel_idx]
        )
    return resized
