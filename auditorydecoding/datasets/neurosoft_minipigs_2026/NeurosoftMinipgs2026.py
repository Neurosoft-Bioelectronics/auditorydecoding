from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Callable, Literal, Optional
from temporaldata import Data, Interval, IrregularTimeSeries

from torch_brain.dataset import Dataset

from auditorydecoding.preprocessing import (
    ChannelSelector,
    PreprocessingPipeline,
    reassemble_segments,
    split_segments,
)


class NeurosoftMinipigs2026(Dataset):
    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        fold_num: Optional[int] = None,
        split_type: Optional[
            Literal["intersubject", "intersession", "intrasession"]
        ] = None,
        task_type: Optional[
            Literal["on_vs_off", "acoustic_stim"]
        ] = "on_vs_off",
        preprocessing: Optional[PreprocessingPipeline] = None,
        dirname: str = "neurosoft_minipigs_2026",
        **kwargs,
    ):
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "channels.id"],
            **kwargs,
        )
        self.fold_num = fold_num
        self.split_type = split_type
        self.task_type = task_type
        self.preprocessing = preprocessing
        self._preprocessed_ecog: dict[str, IrregularTimeSeries] = {}

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(
        self,
        split: str = "train",
        plot: bool = False,
        recording_id: str | None = None,
    ) -> None:
        """Fit preprocessing on *split* data, then transform **all** recordings.

        Preprocessed signals are stored in a side cache so the original lazy
        HDF5 objects in ``_data_objects`` are never replaced.  The override of
        ``get_recording_hook`` swaps the ecog attribute in after the (cheap)
        lazy deep-copy, keeping ``dataset[index]`` fast.

        Parameters
        ----------
        split : str
            Which split to use for fitting (e.g. ``"train"``).
        plot : bool
            If *True*, show 5-second snapshots of the signal at three
            different times across recording segments, after each
            preprocessing step.
        recording_id : str or None
            Which recording to plot.  Defaults to the first recording.
        """
        if self.preprocessing is None:
            return

        # Resolve ChannelSelector indices from channel metadata (once)
        first_data = self._data_objects[self.recording_ids[0]]
        for step in self.preprocessing.steps:
            if isinstance(step, ChannelSelector):
                step.resolve_indices(np.asarray(first_data.channels.type))

        # Collect segments from recordings that contribute to this split
        train_intervals = self.get_sampling_intervals(split=split)
        fit_segments = []
        for rid, interval in train_intervals.items():
            if len(interval.start) == 0:
                continue
            data = self._data_objects[rid]
            fit_segments.extend(split_segments(data.ecog))

        self.preprocessing.fit(fit_segments)

        if plot:
            plot_rid = recording_id or self.recording_ids[0]
            plot_data = self._data_objects[plot_rid]
            from auditorydecoding.plotting import plot_preprocessing_stages

            plot_preprocessing_stages(
                plot_data.ecog,
                self.preprocessing,
                channel_names=np.asarray(plot_data.channels.id),
            )

        # Transform ALL recordings and cache the preprocessed ecog objects.
        # The original _data_objects stay lazy / untouched.
        for rid in self.recording_ids:
            data = self._data_objects[rid]
            segments = split_segments(data.ecog)
            transformed = self.preprocessing.transform(segments)
            self._preprocessed_ecog[rid] = reassemble_segments(
                transformed, data.ecog.domain
            )

    def get_recording_hook(self, data: Data) -> None:
        """Post-process every recording returned by ``get_recording``.

        * Swaps in the preprocessed ecog when available (avoids deep-copying
          the large materialized arrays on every window access).
        * Drops ``splits`` — it is only needed for resolving sampling
          intervals, which now read directly from ``_data_objects``.
        """
        rid = str(data.session.id)
        if rid in self._preprocessed_ecog:
            data.ecog = self._preprocessed_ecog[rid]

        if hasattr(data, "splits"):
            del data.splits

    # ------------------------------------------------------------------
    # Sampling intervals
    # ------------------------------------------------------------------

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
    ):
        if split is None:
            return {
                rid: self._data_objects[rid].domain
                for rid in self.recording_ids
            }
        if split not in ["train", "valid", "test"]:
            raise ValueError(
                "split must be ['train', 'valid', 'test'], or None."
            )
        if self.split_type is None or self.fold_num is None:
            raise ValueError(
                "split_type and fold_num must be set when split is not None."
            )
        if self.task_type not in ["on_vs_off", "acoustic_stim"]:
            raise ValueError(f"Invalid task_type '{self.task_type}'.")

        if self.split_type == "intrasession":
            return self._get_intrasession_intervals(split)
        if self.split_type in ("intersubject", "intersession"):
            return self._get_intersubject_or_intersession_intervals(split)
        raise ValueError(f"Invalid split_type '{self.split_type}'.")

    def _get_intrasession_intervals(
        self, split: Literal["train", "valid", "test"]
    ) -> dict:
        if self.task_type == "on_vs_off":
            key = f"splits.on_vs_off_fold_{self.fold_num}_{split}"
        elif self.task_type == "acoustic_stim":
            key = f"splits.acoustic_stim_fold_{self.fold_num}_{split}"
        else:
            raise ValueError(f"Invalid task_type '{self.task_type}'.")
        return {
            rid: self._data_objects[rid].get_nested_attribute(key)
            for rid in self.recording_ids
        }

    def _get_intersubject_or_intersession_intervals(
        self, split: Literal["train", "valid", "test"]
    ) -> dict:
        if self.split_type == "intersubject":
            assignment_key = (
                f"splits.intersubject_fold_{self.fold_num}_assignment"
            )
        else:
            assignment_key = (
                f"splits.intersession_fold_{self.fold_num}_assignment"
            )

        result = {}
        for rid in self.recording_ids:
            data = self._data_objects[rid]
            # str() guards against h5py returning bytes or numpy.str_
            assignment = str(data.get_nested_attribute(assignment_key))
            if assignment == split:
                if self.task_type == "on_vs_off":
                    result[rid] = data.on_vs_off_trials
                elif self.task_type == "acoustic_stim":
                    result[rid] = data.acoustic_stim_trials
                else:
                    raise ValueError(f"Invalid task_type '{self.task_type}'.")
            else:
                result[rid] = _empty_interval()
        return result


def _empty_interval() -> Interval:
    return Interval(start=np.array([]), end=np.array([]))
