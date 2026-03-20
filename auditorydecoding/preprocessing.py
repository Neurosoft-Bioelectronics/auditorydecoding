from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import scipy.signal as sps
from sklearn.decomposition import PCA
from temporaldata import IrregularTimeSeries, Interval


@dataclass
class Segment:
    """One contiguous recording block extracted from a session's IrregularTimeSeries."""

    signal: np.ndarray  # (n_timepoints, n_channels)
    timestamps: np.ndarray  # (n_timepoints,)


def split_segments(ecog: IrregularTimeSeries) -> list[Segment]:
    """Split an IrregularTimeSeries into per-recording segments using ``ecog.domain``."""
    segments = []
    for i in range(len(ecog.domain.start)):
        start, end = ecog.domain.start[i], ecog.domain.end[i]
        mask = (ecog.timestamps >= start) & (ecog.timestamps <= end)
        segments.append(
            Segment(
                signal=ecog.signal[mask].copy(),
                timestamps=ecog.timestamps[mask].copy(),
            )
        )
    return segments


def reassemble_segments(
    segments: list[Segment], domain: Interval
) -> IrregularTimeSeries:
    """Concatenate transformed segments back into a single IrregularTimeSeries."""
    signal = np.concatenate([s.signal for s in segments], axis=0)
    timestamps = np.concatenate([s.timestamps for s in segments], axis=0)

    new_domain = Interval(
        start=np.array([s.timestamps[0] for s in segments]),
        end=np.array([s.timestamps[-1] for s in segments]),
    )

    return IrregularTimeSeries(
        signal=signal,
        timestamps=timestamps,
        domain=new_domain,
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class PreprocessingStep(ABC):
    """sklearn-style preprocessing step operating on lists of Segments."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def fit(self, segments: list[Segment]) -> PreprocessingStep:
        return self

    @abstractmethod
    def transform(self, segments: list[Segment]) -> list[Segment]: ...

    def fit_transform(self, segments: list[Segment]) -> list[Segment]:
        return self.fit(segments).transform(segments)


# ---------------------------------------------------------------------------
# Concrete steps
# ---------------------------------------------------------------------------


class ChannelSelector(PreprocessingStep):
    """Keep only channels whose ``channels.type`` matches one of the given types.

    Because this step needs access to the channel metadata (not just the signal
    arrays), the matching channel *indices* must be resolved before calling
    ``transform`` by calling :meth:`resolve_indices` once and storing the result.
    """

    def __init__(self, channel_types: list[str]):
        self.channel_types = channel_types
        self._indices: np.ndarray | None = None

    def resolve_indices(self, channel_types_array: np.ndarray) -> None:
        mask = np.isin(channel_types_array, self.channel_types)
        self._indices = np.where(mask)[0]

    def transform(self, segments: list[Segment]) -> list[Segment]:
        if self._indices is None:
            raise RuntimeError(
                "ChannelSelector.resolve_indices() must be called before transform()."
            )
        return [
            Segment(
                signal=seg.signal[:, self._indices],
                timestamps=seg.timestamps,
            )
            for seg in segments
        ]


class ZeroCenter(PreprocessingStep):
    """Subtract the per-channel mean independently within each segment."""

    def transform(self, segments: list[Segment]) -> list[Segment]:
        return [
            Segment(
                signal=seg.signal - seg.signal.mean(axis=0, keepdims=True),
                timestamps=seg.timestamps,
            )
            for seg in segments
        ]


class Whiten(PreprocessingStep):
    """PCA whitening fitted on pooled training segments.

    Uses ``sklearn.decomposition.PCA(whiten=True)`` so each component is
    scaled to unit variance.  ``n_components`` controls optional dimensionality
    reduction (``None`` keeps all components).
    """

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components
        self._pca: PCA | None = None

    def fit(self, segments: list[Segment]) -> Whiten:
        pooled = np.concatenate([s.signal for s in segments], axis=0)
        self._pca = PCA(n_components=self.n_components, whiten=True).fit(pooled)
        return self

    def transform(self, segments: list[Segment]) -> list[Segment]:
        if self._pca is None:
            raise RuntimeError(
                "Whiten.fit() must be called before transform()."
            )
        return [
            Segment(
                signal=self._pca.transform(seg.signal),
                timestamps=seg.timestamps,
            )
            for seg in segments
        ]


class Resample(PreprocessingStep):
    """Resample each segment to ``target_rate`` Hz."""

    def __init__(self, target_rate: float):
        self.target_rate = target_rate

    def transform(self, segments: list[Segment]) -> list[Segment]:
        out = []
        for seg in segments:
            duration = seg.timestamps[-1] - seg.timestamps[0]
            n_new = max(1, int(round(duration * self.target_rate)))
            resampled_signal = sps.resample(seg.signal, n_new, axis=0)
            new_timestamps = np.linspace(
                seg.timestamps[0], seg.timestamps[-1], n_new
            )
            out.append(
                Segment(signal=resampled_signal, timestamps=new_timestamps)
            )
        return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PreprocessingPipeline:
    """Chains multiple :class:`PreprocessingStep` instances."""

    def __init__(self, steps: list[PreprocessingStep]):
        self.steps = steps

    @property
    def step_names(self) -> list[str]:
        return [s.name for s in self.steps]

    def fit(self, segments: list[Segment]) -> PreprocessingPipeline:
        """Fit each step sequentially (each sees the output of the previous)."""
        current = segments
        for step in self.steps:
            current = step.fit_transform(current)
        return self

    def transform(self, segments: list[Segment]) -> list[Segment]:
        for step in self.steps:
            segments = step.transform(segments)
        return segments

    def transform_incremental(
        self, segments: list[Segment]
    ) -> list[list[Segment]]:
        """Return intermediate results after each step (used for plotting).

        Returns a list of length ``len(self.steps) + 1`` where index 0 is the
        raw input and index *i* is the result after step *i-1*.
        """
        stages: list[list[Segment]] = [segments]
        current = segments
        for step in self.steps:
            current = step.transform(current)
            stages.append(current)
        return stages

    def __repr__(self) -> str:
        body = ", ".join(repr(s) for s in self.steps)
        return f"PreprocessingPipeline([{body}])"
