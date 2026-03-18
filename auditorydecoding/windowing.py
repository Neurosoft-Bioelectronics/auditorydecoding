from __future__ import annotations

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
    for index in sampler:
        sample = dataset[index]
        features = feature_extractor(sample.ecog.signal)
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

    X, y = [], []
    for index in sampler:
        rid = index.recording_id
        ecog = preprocessed_ecog[rid]
        ts = ecog.timestamps

        i0 = np.searchsorted(ts, index.start, side="left")
        i1 = np.searchsorted(ts, index.end, side="right")
        signal = ecog.signal[i0:i1]

        X.append(feature_extractor(signal))

        trials = trial_cache[rid]
        mid = (index.start + index.end) / 2
        mask = (trials.start <= mid) & (trials.end >= mid)
        y.append(trials.behavior_labels[mask][0])

    return np.stack(X), np.array(y)
