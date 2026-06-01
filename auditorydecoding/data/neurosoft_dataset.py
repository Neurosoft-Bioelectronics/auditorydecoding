import numpy as np
from pathlib import Path
from typing import Callable, Literal, Optional
from temporaldata import Interval
import random

from torch_brain.dataset import Dataset, MultiChannelDatasetMixin


class NeurosoftDataset(MultiChannelDatasetMixin, Dataset):
    """Neurosoft dataset.

    ``fold_num`` is not used when ``split_type`` is ``'intrasession-causal'``
    (causal splits are single train/valid/test partitions per recording file).
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
        class_balance: Optional[
            Literal["threshold", "downsample", "d-threshold", "percentile"]
        ] = None,
        balance_threshold: Optional[
            int
        ] = 25,
        min_trials: Optional[
            int
        ] = 0,
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
        self.class_balance = class_balance
        self.min_trials = min_trials
        self.percentile_threshold = balance_threshold

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
    ):
        if split is None:
            return {
                rid: self.get_recording(rid).domain
                for rid in self.recording_ids
            }
        if split not in ["train", "valid", "test"]:
            raise ValueError(
                "split must be ['train', 'valid', 'test'], or None."
            )
        if self.split_type is None:
            raise ValueError("split_type must be set when split is not None.")
        if self.task_type not in ["on_vs_off", "acoustic_stim"]:
            raise ValueError(f"Invalid task_type '{self.task_type}'.")

        st = self.split_type
        if st == "intrasession":
            st = "intrasession-block"

        if st == "intrasession-causal":
            intervals = self._get_intrasession_causal_intervals(split)

        if self.fold_num is None:
            raise ValueError(
                "fold_num must be set when split is not None, except for "
                "split_type 'intrasession-causal'."
            )

        if st == "intrasession-block":
            intervals = self._get_intrasession_block_intervals(split)
        if self.split_type in ("intersubject", "intersession"):
            intervals = self._get_intersubject_or_intersession_intervals(split)
        else:
            raise ValueError(f"Invalid split_type '{self.split_type}'.")

        if self.class_balance is not None:
            intervals = self._balance_intervals(intervals)

        return intervals

    def _get_intrasession_block_intervals(
        self, split: Literal["train", "valid", "test"]
    ) -> dict:
        if self.task_type == "on_vs_off":
            key = f"splits.on_vs_off_block_fold_{self.fold_num}_{split}"
        elif self.task_type == "acoustic_stim":
            key = f"splits.acoustic_stim_block_fold_{self.fold_num}_{split}"
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
        
    def _balance_intervals(self, intervals: dict) -> dict:
        """Return a balanced view of *intervals* according to ``self.class_balance``.

        * 'downsample': keep at most *min_count* trials per class, where
          *min_count* is the size of the smallest class present in this
          recording's intervals.  Trials within each class are chosen with a
          fixed random seed for reproducibility.
        * 'threshold': drop every class whose trial count is strictly below
          ``self.min_trials_per_class``.
        * 'd-threshold': first apply the threshold procedure then apply downsample
        """
        
        all_labels = np.concatenate([
            np.asarray(iv.behavior_labels) for iv in intervals.values() if len(iv) > 0 and hasattr(iv, "behavior_labels")
        ])

        if len(all_labels) == 0:
            return intervals
        
        unique_classes, counts = np.unique(all_labels, return_counts=True)

        if self.class_balance == "percentile":
            self.min_trials = np.percentile(counts, self.percentile_threshold)
            valid_classes = set(
                unique_classes[(counts >= self.min_trials)]
            )
            intervals = self._filter_intervals_by_classes(intervals, valid_classes)

        if self.class_balance in ("threshold", "d-threshold"):
            valid_classes = set(
                unique_classes[counts >= self.min_trials]
            )
            intervals = self._filter_intervals_by_classes(intervals, valid_classes)

        if self.class_balance in ("downsample", "d-threshold", "percentile"):
            global_limit = int(counts[counts >= self.min_trials].min())
            rng = np.random.default_rng(self.balance_seed)
            all_indices = {cls: [] for cls in unique_classes}
            for rid, iv in intervals.items():
                if len(iv) > 0 and hasattr(iv, "behavior_labels"):
                    for i, lbl in enumerate(iv.behavior_labels):
                        if lbl in all_indices:
                            all_indices[lbl].append((rid, i))
            kept = {rid: np.zeros(len(iv), dtype=bool) for rid, iv in intervals.items()}
            for cls, idx_list in all_indices.items():
                if len(idx_list) == 0:
                    continue
                n = min(global_limit, len(idx_list))
                chosen = rng.choice(len(idx_list), size=n, replace=False)
                for i in chosen:
                    rid, local_i = idx_list[i]
                    kept[rid][local_i] = True
            intervals = {
                rid: iv.select_by_mask(kept[rid]) for rid, iv in intervals.items()
            }

        return intervals

    def _filter_intervals_by_classes(self, intervals: dict, valid_classes: set) -> dict:
        return {
            rid: (
                iv.select_by_mask(
                    np.isin(np.asarray(iv.behavior_labels), list(valid_classes))
                )
                if len(iv) > 0 and hasattr(iv, "behavior_labels")
                else iv
            )
            for rid, iv in intervals.items()
        }

class NeurosoftMinipigs2026(NeurosoftDataset):
    def __init__(self, **kwargs):
        super().__init__(dirname="neurosoft_minipigs_2026", **kwargs)


class NeurosoftMonkeys2026(NeurosoftDataset):
    def __init__(self, **kwargs):
        super().__init__(dirname="neurosoft_monkeys_2026", **kwargs)


def _empty_interval() -> Interval:
    return Interval(start=np.array([]), end=np.array([]))
