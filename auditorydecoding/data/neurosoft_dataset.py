from collections import defaultdict

import numpy as np
from pathlib import Path
from typing import Callable, Literal, Optional
import random
from torch_brain.data import Interval
from torch_brain.datasets import Dataset, MultiChannelDatasetMixin


class NeurosoftDataset(MultiChannelDatasetMixin, Dataset):
    """Neurosoft dataset.

    ``fold_num`` is not required when ``split_type`` is
    ``'intrasession-causal'`` or ``'intersession'`` (both are single-fold
    partitions).  For ``'intersession'`` it defaults to ``0`` if omitted.
    """

    def __init__(
        self,
        root: str,
        dirname: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        fold_num: Optional[int] = None,
        split_type: Optional[
            Literal[
                "intersubject",
                "intersession",
                "intrasession",
                "intrasession-block",
                "intrasession-causal",
            ]
        ] = None,
        task_type: Optional[
            Literal["on_vs_off", "acoustic_stim"]
        ] = "on_vs_off",
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

        self.valid_splits = ["train", "valid", "test"]
        self.valid_task_types = ["on_vs_off", "acoustic_stim"]

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
    ):
        if split is None:
            return {
                rid: self.get_recording(rid).domain
                for rid in self.recording_ids
            }
        if split not in self.valid_splits:
            raise ValueError(
                "split must be ['train', 'valid', 'test'], or None."
            )
        if self.split_type is None:
            raise ValueError("split_type must be set when split is not None.")
        if self.task_type not in self.valid_task_types:
            raise ValueError(f"Invalid task_type '{self.task_type}'.")

        st = self.split_type
        if st == "intrasession":
            st = "intrasession-block"

        if st == "intrasession-causal":
            intervals = self._get_intrasession_causal_intervals(split)
        else:
            if st == "intersession" and self.fold_num is None:
                fold_num_resolved = 0
            elif self.fold_num is not None:
                fold_num_resolved = self.fold_num
            else:
                raise ValueError(
                    "fold_num must be set when split is not None, except for "
                    "split_type 'intrasession-causal' or 'intersession'."
                )

            if st == "intrasession-block":
                intervals = self._get_intrasession_block_intervals(split, fold_num_resolved)
            elif self.split_type in ("intersubject", "intersession"):
                intervals = self._get_intersubject_or_intersession_intervals(split, fold_num_resolved)
            else:
                raise ValueError(f"Unknown split_type '{self.split_type}'.")
            
        return intervals

    def _get_intrasession_block_intervals(
        self, split: Literal["train", "valid", "test"], fold_num: int
    ) -> dict:
        if self.task_type == "on_vs_off":
            key = f"splits.on_vs_off_block_fold_{fold_num}_{split}"
        elif self.task_type == "acoustic_stim":
            key = f"splits.acoustic_stim_block_fold_{fold_num}_{split}"
        else:
            raise ValueError(f"Invalid task_type '{self.task_type}'.")
        return {
            rid: self.get_recording(rid).get_nested_attribute(key)
            for rid in self.recording_ids
        }

    def _get_intrasession_causal_intervals(
        self, split: Literal["train", "valid", "test"]
    ) -> dict:
        if self.task_type == "on_vs_off":
            key = f"splits.on_vs_off_causal_{split}"
        elif self.task_type == "acoustic_stim":
            key = f"splits.acoustic_stim_causal_{split}"
        else:
            raise ValueError(f"Invalid task_type '{self.task_type}'.")
        return {
            rid: self.get_recording(rid).get_nested_attribute(key)
            for rid in self.recording_ids
        }

    def _get_intersubject_or_intersession_intervals(
        self, split: Literal["train", "valid", "test"], fold_num: int
    ) -> dict:
        if self.split_type == "intersubject":
            assignment_key = f"splits.intersubject_fold_{fold_num}_assignment"
        else:
            assignment_key = f"splits.intersession_fold_{fold_num}_assignment"

        result = {}
        for rid in self.recording_ids:
            data = self.get_recording(rid)
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

    def get_subject_sampling_weights(
        self,
        split: Literal["train", "valid", "test"] = "train",
    ) -> dict[str, float]:
        """Per-recording sampling weights that equalise total weight across subjects.

        Useful when the training set is dominated by one subject (e.g. monkeys
        dataset where sub-01 has 12 sessions vs. 1 each for others). Feed the
        returned weights into a ``WeightedRandomSampler`` or scale the loss.

        Each subject receives equal aggregate weight (``1/N_subjects``),
        distributed uniformly across its active recordings in the requested
        *split*.  Recordings that are empty for the split get weight 0.
        """
        intervals = self.get_sampling_intervals(split)

        subject_recording_counts: dict[str, int] = defaultdict(int)
        rid_to_subject: dict[str, str] = {}
        for rid in self.recording_ids:
            data = self.get_recording(rid)
            sub = str(data.subject.id)
            rid_to_subject[rid] = sub
            if len(intervals[rid]) > 0:
                subject_recording_counts[sub] += 1

        n_subjects = len(subject_recording_counts)
        if n_subjects == 0:
            return {rid: 0.0 for rid in self.recording_ids}

        weights: dict[str, float] = {}
        for rid in self.recording_ids:
            sub = rid_to_subject[rid]
            if sub in subject_recording_counts and len(intervals[rid]) > 0:
                weights[rid] = 1.0 / (
                    subject_recording_counts[sub] * n_subjects
                )
            else:
                weights[rid] = 0.0
        return weights

    def get_recording_hook(self, data):
        # Let the base hook populate defaults first, then enforce Neurosoft readout.
        # This avoids parent logic resetting `multitask_readout` to an empty list.
        super().get_recording_hook(data)
        if not hasattr(data, "config") or data.config is None:
            data.config = {}
        if self.task_type == "on_vs_off":
            data.config["multitask_readout"] = [
                {"readout_id": "neurosoft_on_vs_off"}
            ]
        elif self.task_type == "acoustic_stim":
            data.config["multitask_readout"] = [
                {"readout_id": "neurosoft_acoustic_stim"}
            ]
        else:
            raise ValueError(f"Invalid task_type '{self.task_type}'.")
        

    def set_sampling_intervals(
        self,
        intervals: dict,
        split: Optional[Literal["train", "valid", "test"]] = None,
        ):

        if split is None:
            for rid, interval in intervals.items():
                # TODO: Check if this has a proper setter
                self.get_recording(rid)._domain = interval
            return
        
        if split not in self.valid_splits:
            raise ValueError(
                "split must be ['train', 'valid', 'test], or None."
            )
        
        if split not in self.valid_splits:
            raise ValueError(
                "split must be ['train', 'valid', 'test'], or None."
            )
        if self.split_type is None:
            raise ValueError("split_type must be set when split is not None.")
        if self.task_type not in self.valid_task_types:
            raise ValueError(f"Invalid task_type '{self.task_type}'.")

        st = self.split_type
        if st == "intrasession":
            st = "intrasession-block"

        if st == "intrasession-causal":
            self._set_intrasession_causal_intervals(intervals, split)
        else:
            if self.fold_num is None:
                raise ValueError(
                    "fold_num must be set when split is not None, except for "
                    "split_type 'intrasession-causal'."
                )

            if st == "intrasession-block":
                self._set_intrasession_block_intervals(intervals, split)
            elif self.split_type in ("intersubject", "intersession"):
                self._set_intersubject_or_intersession_intervals(intervals, split)
            else:
                raise ValueError(f"Unknown split_type '{self.split_type}'.")
            
    def _set_intrasession_block_intervals(
        self, intervals: dict, split: Literal["train", "valid", "test"]
    ) -> dict:
        if self.task_type == "on_vs_off":
            key = f"splits.on_vs_off_block_fold_{self.fold_num}_{split}"
        elif self.task_type == "acoustic_stim":
            key = f"splits.acoustic_stim_block_fold_{self.fold_num}_{split}"
        else:
            raise ValueError(f"Invalid task_type '{self.task_type}'.")
        for rid, interval in intervals.items():
            self.get_recording(rid).set_nested_attribute(key, interval)

    def _set_intrasession_causal_intervals(
        self, intervals: dict, split: Literal["train", "valid", "test"]
    ) -> dict:
        if self.task_type == "on_vs_off":
            key = f"splits.on_vs_off_causal_{split}"
        elif self.task_type == "acoustic_stim":
            key = f"splits.acoustic_stim_causal_{split}"
        else:
            raise ValueError(f"Invalid task_type '{self.task_type}'.")
        for rid, interval in intervals.items():
            self.get_recording(rid).set_nested_attribute(key, interval)


    def _set_intersubject_or_intersession_intervals(
        self, intervals: dict, split: Literal["train", "valid", "test"]
    ) -> dict:
        if self.split_type == "intersubject":
            assignment_key = (
                f"splits.intersubject_fold_{self.fold_num}_assignment"
            )
        else:
            assignment_key = (
                f"splits.intersession_fold_{self.fold_num}_assignment"
            )

        for rid, interval in intervals.items():
            data = self.get_recording(rid)
            assignment = str(data.get_nested_attribute(assignment_key))
            if assignment == split:
                if self.task_type == "on_vs_off":
                    data.on_vs_off_trials = interval
                elif self.task_type == "acoustic_stim":
                    data.acoustic_stim_trials
                else:
                    raise ValueError(f"Invalid task_type '{self.task_type}'.")

class NeurosoftMinipigs2026(NeurosoftDataset):
    def __init__(self, **kwargs):
        super().__init__(dirname="neurosoft_minipigs_2026", **kwargs)


class NeurosoftMonkeys2026(NeurosoftDataset):
    def __init__(self, **kwargs):
        super().__init__(dirname="neurosoft_monkeys_2026", **kwargs)


def _empty_interval() -> Interval:
    return Interval(start=np.array([]), end=np.array([]))
