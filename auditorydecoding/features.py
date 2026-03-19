from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.signal as sps


class FeatureExtractor(ABC):
    """Base class for window-level feature extractors.

    Subclasses implement ``__call__`` which maps a signal window of shape
    ``(n_timepoints, n_channels)`` to a flat 1-D feature vector.
    """

    @abstractmethod
    def __call__(self, signal: np.ndarray) -> np.ndarray: ...


class IdentityFeatures(FeatureExtractor):
    """Identity -> ``(n_timepoints, n_channels)``."""

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return signal


class MeanFeatures(FeatureExtractor):
    """Channel-wise mean -> ``(n_channels,)``."""

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return signal.mean(axis=0)


class StdFeatures(FeatureExtractor):
    """Channel-wise standard deviation -> ``(n_channels,)``."""

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return signal.std(axis=0)


class FFTFeatures(FeatureExtractor):
    """Per-channel rfft magnitude, flattened -> ``(n_freq_bins * n_channels,)``."""

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.rfft(signal, axis=0)).flatten()


class FlattenFeatures(FeatureExtractor):
    """Raw signal flattened -> ``(n_timepoints * n_channels,)``."""

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return signal.flatten()


class ResampleFeatures(FeatureExtractor):
    """Resample to ``n_samples`` time points then flatten."""

    def __init__(self, n_samples: int):
        self.n_samples = n_samples

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return sps.resample(signal, self.n_samples, axis=0).flatten()
