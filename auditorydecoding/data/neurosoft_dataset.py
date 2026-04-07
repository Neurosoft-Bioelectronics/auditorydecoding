import numpy as np
from pathlib import Path
from typing import Callable, Literal, Optional
from temporaldata import Interval

from torch_brain.dataset import Dataset


class NeurosoftDataset(Dataset):
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
            return self._get_intrasession_causal_intervals(split)

        if self.fold_num is None:
            raise ValueError(
                "fold_num must be set when split is not None, except for "
                "split_type 'intrasession-causal'."
            )

        if st == "intrasession-block":
            return self._get_intrasession_block_intervals(split)
        if self.split_type in ("intersubject", "intersession"):
            return self._get_intersubject_or_intersession_intervals(split)
        raise ValueError(f"Invalid split_type '{self.split_type}'.")

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


class NeurosoftMinipigs2026(NeurosoftDataset):
    def __init__(self, **kwargs):
        super().__init__(dirname="neurosoft_minipigs_2026", **kwargs)


class NeurosoftMonkeys2026(NeurosoftDataset):
    def __init__(self, **kwargs):
        super().__init__(dirname="neurosoft_monkeys_2026", **kwargs)


def _empty_interval() -> Interval:
    return Interval(start=np.array([]), end=np.array([]))