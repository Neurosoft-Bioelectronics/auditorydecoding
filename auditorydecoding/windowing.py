from __future__ import annotations

from collections import Counter

import numpy as np

from torch_brain.data.sampler import SequentialFixedWindowSampler

from auditorydecoding.features import FeatureExtractor, FlattenFeatures


def extract_windows(
    dataset,
    split: str,
    window_length: float,
    feature_extractor: FeatureExtractor | None = None,
    label_field: str = "on_vs_off_trials",
) -> tuple[np.ndarray, np.ndarray]:
    """Iterate over sampler windows and return ``(X, y)`` numpy arrays.

    When the dataset carries preprocessed ecog data (via ``preprocess()``),
    a fast path uses ``np.searchsorted`` for O(log n) slicing instead of
    going through ``dataset[index]`` which deep-copies and O(n)-scans
    every window.

    Parameters
    ----------
    dataset
        A :class:`~torch_brain.dataset.Dataset` (or compatible) instance.
    split
        Which data split to sample from (``"train"``, ``"valid"``, ``"test"``).
    window_length
        Window duration in seconds.
    feature_extractor
        Callable that maps ``(n_timepoints, n_channels)`` -> 1-D feature
        vector.  Defaults to :class:`FlattenFeatures` if *None*.
    label_field
        Name of the ``Interval`` attribute on each sample that carries
        ``behavior_labels``.
    """
    if feature_extractor is None:
        feature_extractor = FlattenFeatures()

    intervals = dataset.get_sampling_intervals(split=split)
    sampler = SequentialFixedWindowSampler(
        sampling_intervals=intervals,
        window_length=window_length,
        drop_short=True,
    )

    preprocessed = getattr(dataset, "_preprocessed_ecog", {})
    if preprocessed:
        return _extract_fast(
            dataset, sampler, preprocessed, feature_extractor, label_field
        )

    X, y = [], []
    target_num_samples: int | None = None
    for index in sampler:
        sample = dataset[index]
        signal = sample.ecog.signal
        if target_num_samples is None:
            target_num_samples = int(signal.shape[0])
        signal = _ensure_window_length(
            signal, target_num_samples, window_index=index
        )
        features = feature_extractor(signal)
        label = getattr(sample, label_field).behavior_labels[0]
        X.append(features)
        y.append(label)

    return np.stack(X), np.array(y)


def _extract_fast(
    dataset,
    sampler,
    preprocessed_ecog,
    feature_extractor,
    label_field,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch extraction using searchsorted — avoids deepcopy and O(n) masking."""
    trial_cache: dict[str, object] = {}
    for rid in dataset.recording_ids:
        data = dataset._data_objects[rid]
        if hasattr(data, label_field):
            trial_cache[rid] = getattr(data, label_field)

    indexed_windows = []
    window_lengths = []
    for index in sampler:
        rid = index.recording_id
        ecog = preprocessed_ecog[rid]
        ts = ecog.timestamps

        i0 = np.searchsorted(ts, index.start, side="left")
        i1 = np.searchsorted(ts, index.end, side="right")
        indexed_windows.append((index, i0, i1))
        window_lengths.append(int(i1 - i0))

    target_num_samples = _most_common_length(window_lengths)

    X, y = [], []
    for index, i0, i1 in indexed_windows:
        rid = index.recording_id
        ecog = preprocessed_ecog[rid]
        signal = _ensure_window_length(
            ecog.signal[i0:i1], target_num_samples, window_index=index
        )

        X.append(feature_extractor(signal))

        trials = trial_cache[rid]
        mid = (index.start + index.end) / 2
        mask = (trials.start <= mid) & (trials.end >= mid)
        y.append(trials.behavior_labels[mask][0])

    return np.stack(X), np.array(y)


def _most_common_length(lengths: list[int]) -> int:
    if not lengths:
        raise ValueError("No windows were produced by the sampler.")
    return Counter(lengths).most_common(1)[0][0]


def _ensure_window_length(
    signal: np.ndarray,
    target_num_samples: int,
    *,
    window_index,
) -> np.ndarray:
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
            "Encountered an empty window while extracting features. "
            f"recording_id={window_index.recording_id}, "
            f"start={window_index.start}, end={window_index.end}"
        )

    old_axis = np.linspace(0.0, 1.0, n_timepoints, dtype=np.float64)
    new_axis = np.linspace(0.0, 1.0, target_num_samples, dtype=np.float64)
    resized = np.empty((target_num_samples, n_channels), dtype=signal.dtype)
    for channel_idx in range(n_channels):
        resized[:, channel_idx] = np.interp(
            new_axis, old_axis, signal[:, channel_idx]
        )
    return resized
