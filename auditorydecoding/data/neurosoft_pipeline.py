"""Base pipeline class for Neurosoft Bioelectronics datasets.

This module provides the NeurosoftPipeline abstract base class that
handles common functionality for processing datasets from Neurosoft Bioelectronics, including:
- Processing iEEG recordings from Neurosoft Bioelectronics.
- Extracting metadata, channels, and signal information.
- Generating train-valid-test splits.
- Creating a Data object.
- Storing the Data object in an HDF5 file.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Optional, Literal
import warnings
import h5py

import numpy as np
import pandas as pd
from datetime import datetime, timezone

try:
    import mne
    from mne_bids import read_raw_bids, get_entities_from_fname, BIDSPath

    MNE_BIDS_AVAILABLE = True
except ImportError:
    MNE_BIDS_AVAILABLE = False
    mne = None
    read_raw_bids = None
    get_entities_from_fname = None
    BIDSPath = None

from pathlib import Path

from brainsets.utils.split import (
    generate_string_kfold_assignment,
    generate_stratified_folds,
)

from brainsets.utils.bids_utils import (
    fetch_ieeg_recordings,
    check_ieeg_recording_files_exist,
    group_recordings_by_entity,
    build_bids_path,
    get_subject_info,
    load_participants_tsv,
    load_json_sidecar,
)
from brainsets.utils.mne_utils import (
    extract_measurement_date,
    extract_channels,
    concatenate_recordings,
)

from temporaldata import (
    Data,
    Interval,
    IrregularTimeSeries,
)

from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    DeviceDescription,
    SubjectDescription,
    Species,
)
from brainsets import serialize_fn_map

from brainsets.pipeline import BrainsetPipeline

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")

ON_VS_OFF_TO_ID = {
    "off": 0,
    "on": 1,
}

STIM_FREQUENCY_TO_ID = {
    "stim_100Hz": 0,
    "stim_200Hz": 1,
    "stim_300Hz": 2,
    "stim_400Hz": 3,
    "stim_500Hz": 4,
    "stim_650Hz": 5,
    "stim_800Hz": 6,
    "stim_1000Hz": 7,
    "stim_1500Hz": 10,
    "stim_2000Hz": 11,
    "stim_3000Hz": 12,
    "stim_4000Hz": 13,
    "stim_5000Hz": 14,
    "stim_7700Hz": 15,
    "stim_8000Hz": 16,
    "stim_9500Hz": 17,
    "stim_10000Hz": 18,
    "stim_12000Hz": 8,
    "stim_13000Hz": 9,
    "stim_15000Hz": 19,
    "stim_16000Hz": 20,
    "stim_18000Hz": 21,
    "stim_20000Hz": 22,
    "stim_30000Hz": 23,
    "stim_40000Hz": 24,
    "stim_wn": 25,
}

# Per-recording causal split: contiguous chronological blocks by trial count.
# Must sum to 1.0 (train earliest → test latest within each recording).
CAUSAL_TRAIN_RATIO = 0.6
CAUSAL_VALID_RATIO = 0.1
CAUSAL_TEST_RATIO = 0.3
assert np.isclose(
    CAUSAL_TRAIN_RATIO + CAUSAL_VALID_RATIO + CAUSAL_TEST_RATIO,
    1.0,
)


class NeurosoftPipeline(BrainsetPipeline):
    brainset_id: str = None
    """A unique identifier for this brainset. Should be set in subclass or instance."""

    modality = "ieeg"
    """Data modality of the Neurosoft pipeline. Always 'ieeg'."""

    parser = parser
    """ArgumentParser instance for pipeline command-line arguments."""

    skip_sessions: list[str] = []
    """List of session IDs to skip during processing."""

    recording_overlap_policy: str = "shift"  # "shift", "trim", or "drop"
    """Policy when overlapped segments have *different* waveforms (see
    ``_resolve_recording_overlaps``). If the overlap is duplicate data (waveforms match),
    the later recording is trimmed only—this policy is not applied.
    - 'shift': Offset subsequent recordings in time to prevent overlaps.
    - 'trim': Remove overlapping segments from the later recording.
    - 'drop': Drop the later recording if it overlaps.
    """

    @classmethod
    def get_manifest(
        cls, raw_dir: Path, args: Optional[Namespace]
    ) -> pd.DataFrame:
        """
        Generate a manifest DataFrame for iEEG recordings found in a BIDS raw directory.

        Args:
            raw_dir (Path): Path to the root directory containing BIDS-formatted raw data for this brainset.
            args (Optional[Namespace]): Pipeline arguments.

        Returns:
            pd.DataFrame: DataFrame with one row per grouped session/hemisphere. Contains:
                - session_id: (str) Unique session (with hemisphere suffix, e.g. 'sub-01_ses-01_task-acoustic_desc-run1_LH')
                - recording_ids: (List[str]) List of BIDS recording IDs within the session/hemisphere group.

        Raises:
            FileNotFoundError: If the specified BIDS root directory does not exist.
            ValueError: If any recording group contains an unknown or missing hemisphere acquisition.
        """
        if not raw_dir.exists():
            raise FileNotFoundError(
                f"BIDS root directory '{raw_dir}' does not exist."
            )

        recordings = fetch_ieeg_recordings(raw_dir)
        grouped_recordings = group_recordings_by_entity(
            recordings,
            fixed_entities=["subject", "session", "task", "description"],
        )

        grouped_recordings = _regroup_recordings_by_acquisition(
            grouped_recordings
        )

        manifest_list = []
        for session_id, recordings in grouped_recordings.items():
            manifest_list.append(
                {
                    "session_id": session_id,
                    "recording_ids": [
                        recording["recording_id"] for recording in recordings
                    ],
                }
            )

        if not manifest_list:
            raise ValueError(f"No iEEG recordings found in BIDS root {raw_dir}")

        manifest = pd.DataFrame(manifest_list).set_index("session_id")
        return manifest

    def download(self, manifest_item: pd.Series):
        """
        Verifies that all recordings listed in the provided manifest_item are present in the raw data directory.

        Note:
            The 'redownload' flag is ignored since there is no mechanism to download the data from a remote source.
            This function only checks for the existence of the required files in the local raw directory.

        For each recording ID in manifest_item["recording_ids"], checks whether the corresponding iEEG files exist in self.raw_dir.
        If any files are missing, raises FileNotFoundError. If all recordings are present, returns a dictionary containing the session_id
        and list of recording_ids.

        Args:
            manifest_item (pd.Series): Series or object with at least "session_id" and "recording_ids" fields that describe the group of recordings to verify.

        Returns:
            dict: Dictionary with keys:
                - "session_id" (str): Unique session/group identifier.
                - "recording_ids" (list[str]): List of BIDS recording IDs for the group.

        Raises:
            FileNotFoundError: If any of the required recordings are missing from the raw data directory.
        """
        self.update_status("DOWNLOADING")

        recording_ids = manifest_item.recording_ids

        for recording_id in recording_ids:
            if check_ieeg_recording_files_exist(self.raw_dir, recording_id):
                self.update_status(
                    f"Recording {recording_id} already Downloaded"
                )
            else:
                raise FileNotFoundError(f"Recording {recording_id} not found.")

        return {
            "session_id": manifest_item.Index,
            "recording_ids": recording_ids,
        }

    def process(self, download_output: dict) -> Optional[tuple[Data, Path]]:
        """
        Processes a group of recordings for a given session_id and creates a Data object.

        The processing involves:
            1. Loading one or more iEEG recordings from disk using MNE.
            2. Extracting comprehensive metadata including subject, session, and device information.
            3. Extracting channel and signal information per recording.
            4. Applying modality-specific channel selection and formatting.
            5. Extracting behavior intervals for on_vs_off and acoustic_stim tasks.
            6. Generating train-valid-test splits for the on_vs_off and acoustic_stim tasks.
            7. Creating a Data object.

        Args:
            download_output (dict): Output from the `download` step containing keys:
                - "session_id" (str): Unique identifier for the session/group.
                - "recording_ids" (list[str]): Recording IDs to process.

        Returns:
            Optional[Data]: Data object if processing is performed,
                or None if the group was already processed and processing is skipped.
        """
        # Validate recording_overlap_policy
        allowed_policies = {"shift", "trim", "drop"}
        if self.recording_overlap_policy not in allowed_policies:
            raise ValueError(
                f"Invalid recording_overlap_policy '{self.recording_overlap_policy}'. "
                f"Must be one of: {allowed_policies}"
            )

        session_id = download_output.get("session_id")
        entities = get_entities_from_fname(session_id, on_error="raise")
        subject_id = f"sub-{entities['subject']}"
        participants_data = load_participants_tsv(self.raw_dir)

        if session_id in self.skip_sessions:
            self.update_status("Skipping unannotated session")
            return None

        self.processed_dir.mkdir(exist_ok=True, parents=True)
        store_path = self.processed_dir / f"{session_id}.h5"
        if not getattr(self.args, "reprocess", False):
            if store_path.exists():
                self.update_status("Already Processed")
                return None

        # Load all recordings from the same session into a dictionary of raw objects
        self.update_status(f"Loading {self.modality.upper()} recordings")
        recording_ids = download_output.get("recording_ids")
        recordings = load_recordings(self.raw_dir, recording_ids, self.modality)

        # concatenate the recordings
        raw = concatenate_recordings(list(recordings.values()))
        # delete boundary annotations after concatenating the recordings
        _delete_boundary_annotations(raw)

        self.update_status("Extracting Metadata")
        source = "NeurosoftBioelectronics"
        dataset_description = (
            "This dataset contains electrophysiology data from acoustic stimulation at various frequencies. "
            "Each trial consists of 1 second: 0.5 seconds of stimulation followed by 0.5 seconds of rest."
        )

        brainset_description = BrainsetDescription(
            id=self.brainset_id,
            origin_version="0.0.1",
            derived_version="1.0.0",
            source=source,
            description=dataset_description,
        )

        subject_info = get_subject_info(
            subject_id=subject_id, participants_data=participants_data
        )
        subject_description = SubjectDescription(
            id=subject_id,
            species=Species.UNKNOWN,
            age=subject_info["age"],
            sex=subject_info["sex"],
        )

        # extract the measurement date from the first recording in the session
        meas_date = extract_measurement_date(raw)
        session_description = SessionDescription(
            id=session_id, recording_date=meas_date
        )

        device_description = DeviceDescription(id=session_id)

        # Overlap resolution is computed once and reused to avoid
        # repeated warnings and ensure signal and behavior intervals
        # are properly aligned.
        recording_overlap_decisions = _resolve_recording_overlaps(
            recordings,
            overlap_policy=self.recording_overlap_policy,
            on_overlap="warn",
        )

        self.update_status(f"Extracting {self.modality.upper()} Signal")
        signal = extract_signal(
            recordings,
            overlap_decisions=recording_overlap_decisions,
        )

        self.update_status("Building Channels")
        channels = extract_channels(raw)

        self.update_status("Extracting behavior intervals")
        on_vs_off_trials = extract_on_vs_off_trials(
            recordings,
            overlap_decisions=recording_overlap_decisions,
        )
        acoustic_stim_trials = extract_acoustic_stim_trials(
            recordings,
            overlap_decisions=recording_overlap_decisions,
        )

        self.update_status("Generating splits")
        splits = generate_splits(
            subject_id, session_id, on_vs_off_trials, acoustic_stim_trials
        )

        self.update_status("Creating Data Object")
        data = Data(
            brainset=brainset_description,
            subject=subject_description,
            session=session_description,
            device=device_description,
            ecog=signal,
            channels=channels,
            on_vs_off_trials=on_vs_off_trials,
            acoustic_stim_trials=acoustic_stim_trials,
            splits=splits,
            domain=signal.domain,
        )

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


def generate_splits(
    subject_id: str,
    session_id: str,
    on_vs_off_trials: Interval,
    acoustic_stim_trials: Interval,
) -> Data:
    subject_assignments = generate_string_kfold_assignment(
        string_id=subject_id, n_folds=3, val_ratio=0.2, seed=42
    )
    session_assignments = generate_string_kfold_assignment(
        string_id=f"{subject_id}_{session_id}",
        n_folds=3,
        val_ratio=0.2,
        seed=42,
    )
    namespaced_assignments = {
        f"intersubject_fold_{fold_idx}_assignment": assignment
        for fold_idx, assignment in enumerate(subject_assignments)
    }
    namespaced_assignments.update(
        {
            f"intersession_fold_{fold_idx}_assignment": assignment
            for fold_idx, assignment in enumerate(session_assignments)
        }
    )

    # split the 'baseline' trials from the on_vs_off_trials into smaller segments
    on_vs_off_trials = _split_baseline_trials(on_vs_off_trials)

    # get the splits for the on_vs_off_trials
    on_vs_off_splits = {}
    if len(on_vs_off_trials) > 0:
        on_vs_off_folds = generate_stratified_folds(
            intervals=on_vs_off_trials,
            stratify_by="behavior_labels",
            n_folds=3,
            val_ratio=0.2,
            seed=42,
        )

        for k, fold_data in enumerate(on_vs_off_folds):
            on_vs_off_splits[f"on_vs_off_block_fold_{k}_train"] = (
                fold_data.train
            )
            on_vs_off_splits[f"on_vs_off_block_fold_{k}_valid"] = (
                fold_data.valid
            )
            on_vs_off_splits[f"on_vs_off_block_fold_{k}_test"] = fold_data.test

    on_vs_off_causal_train, on_vs_off_causal_valid, on_vs_off_causal_test = (
        _causal_train_valid_test_by_recording(on_vs_off_trials)
    )
    on_vs_off_causal_splits = {
        "on_vs_off_causal_train": on_vs_off_causal_train,
        "on_vs_off_causal_valid": on_vs_off_causal_valid,
        "on_vs_off_causal_test": on_vs_off_causal_test,
    }

    # get the splits for the acoustic_stim_trials
    acoustic_stim_splits = {}
    if len(acoustic_stim_trials) > 0:
        acoustic_stim_folds = generate_stratified_folds(
            intervals=acoustic_stim_trials,
            stratify_by="behavior_labels",
            n_folds=3,
            val_ratio=0.2,
            seed=42,
        )

        for k, fold_data in enumerate(acoustic_stim_folds):
            acoustic_stim_splits[f"acoustic_stim_block_fold_{k}_train"] = (
                fold_data.train
            )
            acoustic_stim_splits[f"acoustic_stim_block_fold_{k}_valid"] = (
                fold_data.valid
            )
            acoustic_stim_splits[f"acoustic_stim_block_fold_{k}_test"] = (
                fold_data.test
            )

    ac_causal_train, ac_causal_valid, ac_causal_test = (
        _causal_train_valid_test_by_recording(acoustic_stim_trials)
    )
    acoustic_stim_causal_splits = {
        "acoustic_stim_causal_train": ac_causal_train,
        "acoustic_stim_causal_valid": ac_causal_valid,
        "acoustic_stim_causal_test": ac_causal_test,
    }

    return Data(
        **namespaced_assignments,
        **on_vs_off_splits,
        **on_vs_off_causal_splits,
        **acoustic_stim_splits,
        **acoustic_stim_causal_splits,
        domain=on_vs_off_trials | acoustic_stim_trials,
    )


def extract_on_vs_off_trials(
    recordings: dict[str, mne.io.Raw],
    overlap_decisions: dict[str, dict],
) -> Interval:
    """
    Extracts 'on' (stimulation) and 'off' (rest and baseline) trials from a collection of Raw objects.

    Args:
        recordings (dict[str, mne.io.Raw]):
            A dictionary mapping recording names to MNE Raw objects from which on/off trial intervals
            will be extracted.
        overlap_decisions (dict[str, dict]): Must be the output of ``_resolve_recording_overlaps`` for the
            same ``recordings``.

    Returns:
        Interval:
            An Interval object containing all detected on and off (stimulation, rest, and baseline) trials.
    """

    def _label_extractor_on_vs_off(desc: str) -> str | None:
        """Extract label for on/off trials."""
        if desc == "rest" or desc == "baseline":
            return "off"
        elif "stim" in desc and (
            "Hz" in desc or "white-noise" in desc or "WhiteNoise" in desc
        ):
            return "on"
        return None

    start_times, end_times, labels, recording_ids = _extract_interval_trials(
        recordings, overlap_decisions, _label_extractor_on_vs_off
    )

    # Map each unique label to an integer (in increasing order)
    label_ids = [ON_VS_OFF_TO_ID[label] for label in labels]

    return Interval(
        start=np.array(start_times),
        end=np.array(end_times),
        timestamps=(np.array(start_times) + np.array(end_times)) / 2,
        behavior_labels=np.array(labels),
        behavior_ids=np.array(label_ids),
        recording_id=np.array(recording_ids, dtype=object),
        timekeys=["start", "end", "timestamps"],
    )


def extract_acoustic_stim_trials(
    recordings: dict[str, mne.io.Raw],
    overlap_decisions: dict[str, dict],
) -> Interval:
    """Extracts the acoustic stimulation trials across multiple raw recordings.

    Keeps tone stimuli (``stim`` + ``Hz`` in the description) with labels ``stim_<frequency>Hz``,
    and white-noise stimuli (``stim`` + ``white-noise``) with label ``stim_wn``.

    Args:
        recordings (dict[str, mne.io.Raw]):
            A dictionary mapping recording names to MNE Raw objects from which acoustic stimulation trial intervals
            will be extracted.
        overlap_decisions (dict[str, dict]): Must be the output of ``_resolve_recording_overlaps`` for the
            same ``recordings``.

    Returns:
        Interval:
            An Interval object containing start, end, and label information for each detected
            acoustic stimulation trial, along with label IDs.
    """

    def _label_extractor_acoustic_stim(desc: str) -> str | None:
        """Extract label for acoustic stimulation trials."""
        if ("stim" in desc and "white-noise" in desc) or (
            "stim" in desc and "WhiteNoise" in desc
        ):
            return "stim_wn"
        elif "stim" in desc and "Hz" in desc:
            frequency = _extract_stim_frequency(desc)
            return f"stim_{frequency}Hz"
        return None

    start_times, end_times, labels, recording_ids = _extract_interval_trials(
        recordings, overlap_decisions, _label_extractor_acoustic_stim
    )

    # Map each unique label to an integer (in increasing order)
    label_ids = [STIM_FREQUENCY_TO_ID[label] for label in labels]

    return Interval(
        start=np.array(start_times),
        end=np.array(end_times),
        timestamps=(np.array(start_times) + np.array(end_times)) / 2,
        behavior_labels=np.array(labels),
        behavior_ids=np.array(label_ids),
        recording_id=np.array(recording_ids, dtype=object),
        timekeys=["start", "end", "timestamps"],
    )


def extract_signal(
    recordings: dict[str, mne.io.Raw],
    overlap_decisions: dict[str, dict],
) -> IrregularTimeSeries:
    """Extracts the signal from a session of recordings.

    Timestamps are computed relative to the start of the first recording (0),
    with each subsequent recording offset by the actual wall-clock elapsed time
    since the first recording started (via meas_date). This preserves gaps
    between recordings in the timeline.

    Args:
        recordings (dict[str, mne.io.Raw]):
            A dictionary mapping recording names to MNE Raw objects from which signal will be extracted.
        overlap_decisions (dict[str, dict]): Must be the output of ``_resolve_recording_overlaps`` for the
            same ``recordings``.

    Returns:
        IrregularTimeSeries: The extracted signal with relative timestamps.
    """
    signal = []
    timestamps = []
    domain_start = []
    domain_end = []

    for recording_id, raw in recordings.items():
        decision = overlap_decisions[recording_id]

        # Skip dropped recordings
        if not decision["keep"]:
            continue

        # Extract signal data (possibly trimmed)
        start_sample = decision["trim_start_samples"]
        if start_sample >= raw.n_times:
            # Entire recording was trimmed due to overlap handling.
            warnings.warn(
                f"Recording ID: {recording_id} was fully trimmed due to overlap handling."
            )
            continue

        signal_data = raw.get_data(start=start_sample, stop=raw.n_times)
        signal.append(signal_data)

        # Compute timestamps with offset and trimming
        recording_times = raw.times[start_sample:].astype(np.float64)
        ts = recording_times + decision["time_offset"]
        timestamps.append(ts[:, np.newaxis])

        domain_start.append(ts[0])
        domain_end.append(ts[-1])

    if len(signal) == 0:
        raise ValueError(
            "No signal data available after overlap handling; all recordings were dropped or fully trimmed."
        )

    return IrregularTimeSeries(
        signal=np.hstack(signal).T,
        timestamps=np.vstack(timestamps).squeeze(),
        domain=Interval(
            start=np.array(domain_start),
            end=np.array(domain_end),
        ),
    )


def load_recordings(
    raw_dir: Path,
    recording_ids: list[str],
    modality: str,
) -> dict[str, mne.io.Raw]:
    """Load recordings belonging to a given session from the raw directory.

    Args:
        raw_dir (Path): The raw directory to load the recordings from.
        session_id (str): The session ID to load the recordings for.
        recording_ids (list[str]): The list of recording IDs to load.
        modality (str): The modality of the recordings to load (e.g. 'ieeg').
    Returns:
        dict[str, mne.io.Raw]: The dictionary of loaded recordings sorted
        by measurement date.
    """
    session = {}
    for recording_id in recording_ids:
        bids_path = build_bids_path(raw_dir, recording_id, modality)
        raw = read_raw_bids(
            bids_path,
            on_ch_mismatch="reorder",
            verbose="CRITICAL",
        )

        # check if the recording has annotations
        if not raw.annotations or len(raw.annotations) == 0:
            if "Baseline" in recording_id:
                warnings.warn(
                    f"No annotations found in baseline recording {recording_id}. Adding baseline annotations."
                )
                _add_baseline_annotations(raw)
            else:
                warnings.warn(
                    f"No annotations found in recording {recording_id}. Skipping."
                )
                continue
        else:
            # add rest annotations if not present
            if "rest" not in np.unique(raw.annotations.description):
                _add_rest_annotations(raw)

        # update meas_date to the original recording timestamp
        meas_date = datetime.fromisoformat(
            load_json_sidecar(bids_path)["OriginalRecordingTimestamp"]
        )
        if meas_date.tzinfo is None:
            meas_date = meas_date.replace(tzinfo=timezone.utc)
        raw.set_meas_date(meas_date)

        # verify that all baseline annotations have the same description
        _verify_baseline_annotations(raw)

        session[recording_id] = raw

    session = _sort_recordings(session)

    return session


def _regroup_recordings_by_acquisition(
    grouped_recordings: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """
    Groups a dictionary of recordings by acquisition, based on the 'acquisition' entity in the recording IDs.

    Args:
        grouped_recordings (dict[str, list[dict]]):
            A dictionary where keys are group IDs and values are lists of recording dicts.
            Each dict must have the key 'recording_id'.

    Returns:
        dict[str, list[dict]]:
            A new dictionary where each key represents a unique combination of group ID and acquisition (e.g. 'group_LH' or 'group_LHanest'),
            and the values are lists of recording dicts belonging to that acquisition group.

    Raises:
        ValueError: If any recording's acquisition cannot be determined from the 'acquisition' entity.
    """
    # TODO: This is a hack to group by acquisition. This should be done in a more elegant way.
    # The right way would be to use a different 'run' number for each stimulation frequency
    # and use 'acq' to denote acquisition. Stimulation frequency values are included in the *events.tsv file.
    recordings_grouped_by_acquisition = {}
    for group_id, recordings in grouped_recordings.items():
        new_acquisitions = {}
        for recording in recordings:
            recording_id = recording["recording_id"]
            acquisition = get_entities_from_fname(
                recording_id, on_error="raise"
            ).get("acquisition", "")

            new_acquisition = ""
            if acquisition and "L" in acquisition:
                new_acquisition += "LH"
            elif acquisition and "R" in acquisition:
                new_acquisition += "RH"

            if acquisition and "anest" in acquisition:
                new_acquisition += "anest"

            if not new_acquisition:
                new_acquisition = acquisition

            new_acquisitions.setdefault(new_acquisition, []).append(recording)

        # If there's more than one hemisphere in this group, split up
        if len(new_acquisitions) > 0:
            for new_acq, recordings in new_acquisitions.items():
                entities = get_entities_from_fname(group_id, on_error="raise")
                entities["acquisition"] = new_acq
                # The task entity is modified to make the session name
                # appear shorter in the manifest. The task entity of the recordings
                # is left unchanged.
                entities["task"] = "AcousStim"
                new_group_id = BIDSPath(**entities).basename
                recordings_grouped_by_acquisition[new_group_id] = recordings

    if len(recordings_grouped_by_acquisition) == 0:
        return grouped_recordings
    return recordings_grouped_by_acquisition


def _sort_recordings(
    recordings: dict[str, mne.io.Raw],
) -> dict[str, mne.io.Raw]:
    """Sorts the recordings by their measurement date.

    Args:
        recordings (dict[str, mne.io.Raw]): The recordings to sort.

    Returns:
        dict[str, mne.io.Raw]: Recordings as a dictionary, sorted by measurement date.
    """
    sorted_items = sorted(
        recordings.items(), key=lambda x: x[1].info["meas_date"]
    )
    return dict(sorted_items)


def _check_overlap_waveform_similarity(
    previous_recording: mne.io.Raw,
    current_recording: mne.io.Raw,
    overlap_duration_seconds: float,
    correlation_threshold: float = 0.9,
    relative_rmse_threshold: float = 0.2,
) -> tuple[bool, dict[str, float]]:
    """Compare waveforms in overlapping time windows between consecutive recordings.

    Extracts the tail segment of the previous recording and the head segment of the
    current recording (both spanning the overlap duration) and computes similarity metrics.

    Args:
        previous_recording (mne.io.Raw): The earlier recording in chronological order.
        current_recording (mne.io.Raw): The later recording in chronological order.
        overlap_duration_seconds (float): Duration of the overlap in seconds (must be positive).
        correlation_threshold (float): Minimum mean channel correlation (0-1) for segments to be
            considered similar. Default is 0.9.
        relative_rmse_threshold (float): Maximum relative RMSE (normalized by signal RMS) for
            segments to be considered similar. Default is 0.2.

    Returns:
        tuple[bool, dict[str, float]]: A tuple containing:
            - is_similar (bool): True if both correlation and RMSE thresholds are met.
            - metrics (dict[str, float]): Dictionary with keys:
                - "mean_correlation": Average Pearson correlation across channels (0-1 range).
                - "relative_rmse": RMSE normalized by previous segment RMS.
                - "overlap_sample_count": Number of overlapping samples compared.
    """
    # No overlap means segments are trivially identical
    if overlap_duration_seconds <= 0:
        return True, {
            "mean_correlation": 1.0,
            "relative_rmse": 0.0,
            "overlap_sample_count": 0.0,
        }

    # Compute overlap sample count (minimum across both sampling rates)
    if not previous_recording.info["sfreq"] == current_recording.info["sfreq"]:
        raise ValueError(
            "Checking waveform overlap requires that previous and current recordings have the same sampling rate. "
            f"Found: previous = {previous_recording.info['sfreq']}, current = {current_recording.info['sfreq']}. "
            "This is not supported yet."
        )
    sfreq = float(previous_recording.info["sfreq"])
    overlap_sample_count = int(round(overlap_duration_seconds * sfreq))
    overlap_sample_count = min(
        overlap_sample_count,
        previous_recording.n_times,
        current_recording.n_times,
    )

    if overlap_sample_count <= 1:
        return True, {
            "mean_correlation": 1.0,
            "relative_rmse": 0.0,
            "overlap_sample_count": float(overlap_sample_count),
        }

    # Extract tail of previous recording and head of current recording
    previous_tail_data = previous_recording.get_data(
        start=previous_recording.n_times - overlap_sample_count,
        stop=previous_recording.n_times,
    )
    current_head_data = current_recording.get_data(
        start=0,
        stop=overlap_sample_count,
    )

    # Match channel count between previous and current recordings
    if previous_tail_data.shape[0] != current_head_data.shape[0]:
        raise ValueError(
            "Checking waveform overlap requires that previous and current recordings have the same number of channels. "
            f"Found: previous = {previous_tail_data.shape[0]}, current = {current_head_data.shape[0]}. "
            "This is not supported yet."
        )
    num_channels = previous_tail_data.shape[0]

    # Compute relative RMSE (normalized by signal RMS of previous segment)
    difference = previous_tail_data - current_head_data
    rmse = float(np.sqrt(np.mean(difference**2)))
    signal_rms = float(np.sqrt(np.mean(previous_tail_data**2)))
    relative_rmse = rmse / (signal_rms + 1e-12)

    # Compute per-channel correlation, then average
    channel_correlations = []
    for channel_index in range(num_channels):
        previous_channel = previous_tail_data[channel_index]
        current_channel = current_head_data[channel_index]

        previous_std = np.std(previous_channel)
        current_std = np.std(current_channel)

        # Skip flat channels (no variation)
        if previous_std < 1e-12 or current_std < 1e-12:
            continue

        correlation = float(
            np.corrcoef(previous_channel, current_channel)[0, 1]
        )
        channel_correlations.append(correlation)

    mean_correlation = (
        float(np.mean(channel_correlations)) if channel_correlations else 0.0
    )

    # Determine similarity based on both thresholds
    is_similar = (
        mean_correlation >= correlation_threshold
        and relative_rmse <= relative_rmse_threshold
    )

    return is_similar, {
        "mean_correlation": mean_correlation,
        "relative_rmse": relative_rmse,
        "overlap_sample_count": float(overlap_sample_count),
    }


def _resolve_recording_overlaps(
    recordings: dict[str, mne.io.Raw],
    overlap_policy: Literal["shift", "trim", "drop"] = "shift",
    on_overlap: Literal["ignore", "warn", "raise"] = "warn",
) -> dict[str, dict]:
    """Resolve overlaps between consecutive recordings.

    When two recordings overlap in time, the overlapped tail of the earlier file and
    head of the later file are compared. If waveforms match (duplicate segment), the
    overlap policy is ignored: only the earlier segment is kept and the duplicate
    prefix is trimmed from the later recording. If waveforms differ, ``overlap_policy``
    (shift / trim / drop) is applied.

    Overlap policy options:
        - "shift":  Later recordings are shifted forward in time to eliminate the overlap.
        - "trim":   The overlapping segment at the beginning of the later recording is trimmed.
        - "drop":   The entire later recording is dropped if its start overlaps the previous recording.

    Args:
        recordings (dict[str, mne.io.Raw]): Dictionary of recording_id -> Raw objects (pre-sorted by meas_date).
        overlap_policy (Literal["shift", "trim", "drop"]): One of "shift", "trim", or "drop" (used only when waveforms differ).
        on_overlap (Literal["ignore", "warn", "raise"]): How to handle overlap.
            - "ignore": Silently ignore waveform overlap.
            - "warn": Issue a warning and continue (default).
            - "raise": Raise an error and stop the pipeline.

    Returns:
        dict[str, dict]: Dictionary mapping recording_id to overlap decision dict with keys:
            - recording_id: str
            - keep: bool
            - time_offset: float
            - trim_start_samples: int
            - time_overlap_detected: bool
            - overlap_duration_seconds: float
            - waveform_metrics: Optional[dict[str, float]]
    """
    decisions: dict[str, dict] = {}
    first_meas_date = None
    # Accumulates timeline shifts applied by the "shift" policy so extract_signal
    # can build non-overlapping timestamps, while overlap detection itself stays
    # anchored to original recording timepoints.
    cumulative_time_shift = 0.0
    previous_end_time_original = None
    previous_recording_id = None
    previous_raw = None

    for recording_id, raw in recordings.items():
        meas_date = raw.info["meas_date"]
        if first_meas_date is None:
            first_meas_date = meas_date

        # Compute initial offset from measurement date
        base_offset = (meas_date - first_meas_date).total_seconds()
        current_offset = base_offset + cumulative_time_shift

        # Original timeline (from meas_date), used for overlap detection only.
        start_time_original = float(raw.times[0] + base_offset)
        end_time_original = float(raw.times[-1] + base_offset)

        time_overlap_detected = False
        overlap_duration = 0.0
        waveform_metrics = None
        keep = True
        trim_start_samples = 0
        is_similar = False

        # Check for overlap with previous recording
        if (
            previous_end_time_original is not None
            and start_time_original < previous_end_time_original
        ):
            time_overlap_detected = True
            overlap_duration = float(
                previous_end_time_original - start_time_original
            )

            # Check for waveform overlap between previous and current recordings
            is_similar, waveform_metrics = _check_overlap_waveform_similarity(
                previous_recording=previous_raw,
                current_recording=raw,
                overlap_duration_seconds=overlap_duration,
            )

            metrics_suffix = ""
            if waveform_metrics is not None:
                metrics_suffix = (
                    f" Overlap waveform check: "
                    f"corr={waveform_metrics['mean_correlation']:.4f}, "
                    f"rmse={waveform_metrics['relative_rmse']:.4f}."
                )

            if is_similar:
                # Duplicate segment: keep earlier recording; trim duplicate prefix from later
                sampling_rate = float(raw.info["sfreq"])
                trim_start_samples = (
                    int(np.ceil(overlap_duration * sampling_rate)) + 1
                )
                trim_start_samples = min(trim_start_samples, raw.n_times)
                actual_trim_duration = trim_start_samples / sampling_rate

                if on_overlap == "raise":
                    raise ValueError(
                        f"Detected recording overlap between {previous_recording_id} "
                        f"and {recording_id}: {overlap_duration:.4f}s. "
                    )
                elif on_overlap == "warn":
                    warnings.warn(
                        f"Detected recording overlap between {previous_recording_id} "
                        f"and {recording_id}: {overlap_duration:.4f}s. Overlapped segments match; trimming duplicate prefix from the "
                        f"later recording ({trim_start_samples} samples, {actual_trim_duration:.4f}s).{metrics_suffix}"
                    )
                elif on_overlap == "ignore":
                    pass
            else:
                # Different waveforms: apply configured overlap policy
                base_msg = (
                    f"Detected recording overlap between {previous_recording_id} and {recording_id}: "
                    f"{overlap_duration:.4f}s. Overlapped segments differ.{metrics_suffix} "
                )

                # Choose message suffix based on policy
                if overlap_policy == "shift":
                    shift_amount = overlap_duration + (
                        1.0 / float(raw.info["sfreq"])
                    )
                    cumulative_time_shift += shift_amount
                    current_offset += shift_amount
                    action_msg = "Applying 'shift' policy."
                elif overlap_policy == "trim":
                    sampling_rate = float(raw.info["sfreq"])
                    trim_start_samples = (
                        int(np.ceil(overlap_duration * sampling_rate)) + 1
                    )
                    trim_start_samples = min(trim_start_samples, raw.n_times)
                    actual_trim_duration = trim_start_samples / sampling_rate
                    action_msg = f"Applying 'trim' policy: discarding {trim_start_samples} leading samples ({actual_trim_duration:.4f}s)."
                elif overlap_policy == "drop":
                    keep = False
                    action_msg = (
                        f"Applying 'drop' policy: dropping {recording_id}."
                    )

                if on_overlap == "raise":
                    raise ValueError(base_msg + action_msg)
                elif on_overlap == "warn":
                    warnings.warn(base_msg + action_msg)
                elif on_overlap == "ignore":
                    pass

        # Store decision
        decisions[recording_id] = {
            "recording_id": recording_id,
            "keep": keep,
            "time_overlap_detected": time_overlap_detected,
            "overlap_duration_seconds": overlap_duration,
            "waveform_metrics": waveform_metrics,
            "waveform_overlap_detected": is_similar,
            "time_offset": current_offset,
            "trim_start_samples": trim_start_samples,
        }

        # Update state for next iteration (only if we kept this recording)
        if keep:
            previous_end_time_original = end_time_original
            previous_recording_id = recording_id
            previous_raw = raw

    return decisions


def _verify_baseline_annotations(raw: mne.io.Raw) -> None:
    """
    Verifies that all baseline annotations have the same 'baseline' description.

    This function inspects all annotations in the provided MNE Raw object and checks for any annotation whose description contains
    the substring 'baseline', but is not exactly equal to 'baseline' (for example, descriptions like 'baseline_1', 'pre-baseline', etc.).
    Such annotations are automatically renamed to use the standardized label 'baseline' as their description.

    This standardization is necessary because some of the recordings have mixed acoustic stimulation and baseline trials, as opposed
    to having an entire recording be a baseline trial as is the case for most of the baseline trials in the dataset.

    Args:
        raw (mne.io.Raw): The Raw object whose annotations should be checked and standardized in-place.
    """
    verified_descriptions = [
        "baseline"
        if "baseline" in annotation["description"]
        and annotation["description"] != "baseline"
        else annotation["description"]
        for annotation in raw.annotations
    ]
    raw.annotations.description = verified_descriptions


def _add_baseline_annotations(raw: mne.io.Raw) -> None:
    """
    Adds a 'baseline' annotation covering the entire duration of a Raw object.

    This function is intended for use on MNE Raw objects that correspond to baseline recordings and
    that lack explicit 'baseline' annotations in the annotations file. It creates a single annotation
    labeled 'baseline' that spans the entire time range from the first to the last time point in the Raw object.
    The annotation is added in addition to any existing annotations present in the raw data.

    Args:
        raw (mne.io.Raw): The Raw object to which the baseline annotation will be added in place.
    """
    onset = np.array(raw.times[0])
    duration = np.array(raw.times[-1] - raw.times[0])
    description = np.array(["baseline"])

    baseline_annot = mne.Annotations(
        onset=onset,
        duration=duration,
        description=description,
        orig_time=raw.annotations.orig_time,
    )

    raw.set_annotations(raw.annotations + baseline_annot)


def _add_rest_annotations(raw: mne.io.Raw) -> None:
    """
    Adds 'rest' annotations to a Raw object by inferring rest intervals between existing annotations.

    This function scans the annotations currently present in the MNE Raw object (typically corresponding to acoustic stimulation trials
    or baseline periods) and identifies the gaps between them. Each of these gaps is assumed to represent a rest period
    (i.e., the animal is not being stimulated and is at rest). For every such interval, a new annotation with type 'rest' is added,
    with onset equal to the end of the previous annotation and duration extending up to the onset of the next annotation.

    This ensures every period in the recording is assigned a behavioral context, allowing downstream analyses to distinguish
    between rest and stimulation/baseline states, even if explicit 'rest' annotations were not originally present in the data.

    Args:
        raw (mne.io.Raw): The Raw object to add inferred 'rest' annotations to. Its existing annotations will be augmented in place.
    """
    annot = raw.annotations
    order = np.argsort(annot.onset)

    on = annot.onset[order]
    off = on + annot.duration[order]

    rest_onset = off[:-1]
    rest_duration = on[1:] - off[:-1]

    mask = rest_duration > 0
    rest_onset, rest_duration = rest_onset[mask], rest_duration[mask]

    rest_annot = mne.Annotations(
        onset=rest_onset,
        duration=rest_duration,
        description=["rest"] * len(rest_onset),
        orig_time=raw.annotations.orig_time,
    )

    raw.set_annotations(raw.annotations + rest_annot)


def _delete_boundary_annotations(raw: mne.io.Raw) -> None:
    """Deletes the boundary annotations from a raw object.

    Args:
        raw (mne.io.Raw): The raw object to delete the boundary annotations from.
    """
    boundary_annotations = ["BAD boundary", "EDGE boundary"]
    annot = raw.annotations

    # get the indices of boundary annotations to be deleted using numpy vectorization
    description = np.asarray(annot.description)
    mask = np.isin(description, boundary_annotations)
    idx_to_be_deleted = np.where(mask)[0].tolist()

    # delete the boundary annotations
    raw.annotations.delete(idx_to_be_deleted)


def _extract_stim_frequency(description: str) -> str:
    """Extracts the stimulation frequency as an integer from a string containing 'Hz' or 'kHz'.

    Args:
        description (str): Text containing the frequency (e.g. "stim_120Hz" or "stim_120kHz")

    Returns:
        int: The frequency value as an integer (e.g., 120 or 120000)

    Raises:
        ValueError: If no frequency is found before 'Hz' or 'kHz'.
    """
    import re

    match = re.search(r"(\d+(?:\.\d+)?)\s*Hz", description, re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r"(\d+(?:\.\d+)?)\s*kHz", description, re.IGNORECASE)
    if match:
        return int(match.group(1)) * 1000

    raise ValueError(f"No frequency found in description: {description}")


def _extract_interval_trials(
    recordings: dict[str, mne.io.Raw],
    overlap_decisions: dict[str, dict],
    label_extractor: callable,
) -> tuple[list, list, list, list]:
    """Extract interval trials from recordings with flexible label filtering.

    This helper centralizes the common logic for extracting trial intervals from annotations:
    trimming, overlap clipping, and time-offset adjustment. Label assignment and filtering
    is delegated to the caller via a callback function.

    Args:
        recordings (dict[str, mne.io.Raw]):
            A dictionary mapping recording names to MNE Raw objects.
        overlap_decisions (dict[str, dict]):
            Must be the output of ``_resolve_recording_overlaps`` for the same ``recordings``.
        label_extractor (callable):
            A function that takes an annotation description string and returns either:
            - A label string (str) to include the annotation in the output
            - None to skip the annotation

    Returns:
        tuple[list, list, list, list]:
            Four lists: (start_times, end_times, labels, recording_ids) in iteration order.
            These are ready to be converted to numpy arrays and used in an Interval.
    """
    start_times = []
    end_times = []
    labels = []
    recording_ids: list[str] = []

    for recording_id, raw in recordings.items():
        decision = overlap_decisions[recording_id]

        # Skip dropped recordings
        if not decision["keep"]:
            continue

        # Determine start sample index based on trim policy
        start_sample = decision["trim_start_samples"]
        if start_sample >= raw.n_times:
            # Entire recording was trimmed due to overlap handling.
            continue
        trim_start_time = float(raw.times[start_sample])

        for annotation in raw.annotations:
            annotation_start = annotation["onset"]
            annotation_duration = annotation["duration"]

            # Skip annotations that fall entirely within trimmed region
            if annotation_start + annotation_duration <= trim_start_time:
                continue

            # Adjust annotation start if it begins in trimmed region
            if annotation_start < trim_start_time:
                overlap = trim_start_time - annotation_start
                annotation_start = trim_start_time
                annotation_duration -= overlap

            # Skip invalid annotations after trimming
            if annotation_duration <= 0:
                continue

            start_time = annotation_start + decision["time_offset"]
            end_time = start_time + annotation_duration

            if len(end_times) > 0 and start_time < end_times[-1]:
                # the previous trial goes into the next one
                # this happens because of numerical precision issues
                assert end_times[-1] - start_time < 0.1, (
                    f"found overlap between trials: start time of trial i: {start_time}, end time of trial i-1: {end_times[-1]}"
                )

                # we can clip the end time of the last trial
                end_times[-1] = start_time

            desc = annotation["description"]
            label = label_extractor(desc)
            if label is not None:
                start_times.append(start_time)
                end_times.append(end_time)
                recording_ids.append(recording_id)
                labels.append(label)

    return start_times, end_times, labels, recording_ids


def _split_baseline_trials(on_vs_off_trials: Interval) -> Interval:
    """
    Splits the 'baseline' trials in the on_vs_off_trials Interval into smaller segments.

    Baseline trials are defined as those with behavior_labels == "off" and duration >= 0.5s.
    These longer baseline trials are subdivided into 0.5s segments.
    The subdivided baseline trials are then combined with the original "on" trials and returned
    as a new Interval.

    Args:
        on_vs_off_trials (Interval): An Interval containing on/off behavioral trial segments.

    Returns:
        Interval: An Interval object where:
            - All "on" (stimulation) trials are preserved as in the input.
            - All "off" trials that are ≤ 0.5 seconds long ("rest" trials) are preserved as in the input.
            - All "off" trials longer than 0.5 seconds ("baseline" trials) are subdivided into contiguous 0.5-second segments.
            - The returned Interval contains all of the above trials merged together, retaining their respective behavior labels.
    """
    # get the on (stimulation) trials
    on_trials = on_vs_off_trials.select_by_mask(
        on_vs_off_trials.behavior_labels == "on"
    )

    # get the rest trials (off trials shorter or equal to 0.5s)
    rest_trials = on_vs_off_trials.select_by_mask(
        np.logical_and(
            on_vs_off_trials.behavior_labels == "off",
            (on_vs_off_trials.end - on_vs_off_trials.start) <= 0.5,
        )
    )

    # get the baseline trials (off trials longer than 0.5s)
    eps = 1e-4  # account for numerical precision issues
    baseline_trials = on_vs_off_trials.select_by_mask(
        np.logical_and(
            on_vs_off_trials.behavior_labels == "off",
            (on_vs_off_trials.end - on_vs_off_trials.start) > (0.5 + eps),
        )
    )

    # split the baseline trials into 0.5s segments
    split_baseline_trials = baseline_trials.subdivide(0.5, drop_short=False)

    # Create a new Interval with the the split baseline trials
    start_times = np.concatenate(
        [
            on_trials.start,
            rest_trials.start,
            split_baseline_trials.start,
        ]
    )

    end_times = np.concatenate(
        [
            on_trials.end,
            rest_trials.end,
            split_baseline_trials.end,
        ]
    )

    behavior_labels = np.concatenate(
        [
            on_trials.behavior_labels,
            rest_trials.behavior_labels,
            split_baseline_trials.behavior_labels,
        ]
    )

    behavior_ids = np.concatenate(
        [
            on_trials.behavior_ids,
            rest_trials.behavior_ids,
            split_baseline_trials.behavior_ids,
        ]
    )

    recording_ids = np.concatenate(
        [
            on_trials.recording_id,
            rest_trials.recording_id,
            split_baseline_trials.recording_id,
        ]
    )

    new_kwargs = {
        "start": start_times,
        "end": end_times,
        "behavior_labels": behavior_labels,
        "behavior_ids": behavior_ids,
        "timestamps": (start_times + end_times) / 2,
        "timekeys": ["start", "end", "timestamps"],
        "recording_id": recording_ids,
    }
    new_on_vs_off_trials = Interval(**new_kwargs)
    new_on_vs_off_trials.sort()

    return new_on_vs_off_trials


def _causal_counts_per_recording(n: int) -> tuple[int, int, int]:
    """Return (n_train, n_valid, n_test) trial counts for one recording.

    The number of trials in the train, valid, and test sets is determined by
    the CAUSAL_TRAIN_RATIO, CAUSAL_VALID_RATIO, and CAUSAL_TEST_RATIO constants.

    Args:
        n (int): The number of trials in the recording.

    Returns:
        tuple[int, int, int]: The number of trials in the train, valid, and test sets.
    """
    if n == 0:
        return 0, 0, 0
    if n < 3:
        return n, 0, 0
    nt = int(n * CAUSAL_TRAIN_RATIO)
    nv = int(n * CAUSAL_VALID_RATIO)
    nte = n - nt - nv
    if nte < 0:
        nte = 0
        nv = max(0, n - nt - nte)
        if nt + nv > n:
            nt = max(0, n - nv)
    return nt, nv, nte


def _causal_train_valid_test_by_recording(
    intervals: Interval,
) -> tuple[Interval, Interval, Interval]:
    """Chronological train / valid / test by trial count within each recording.

    Args:
        intervals (Interval): The intervals to split.

    Returns:
        tuple[Interval, Interval, Interval]: The train, valid, and test intervals.
    """
    if len(intervals) == 0:
        return intervals, intervals, intervals

    if not hasattr(intervals, "recording_id"):
        raise ValueError("intervals must have recording_id for causal splits.")

    rec = np.asarray(intervals.recording_id, dtype=object)
    train_masks: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    test_masks: list[np.ndarray] = []

    for rid in np.unique(rec):
        idx = np.where(rec == rid)[0]
        order = idx[np.argsort(intervals.start[idx])]
        n = len(order)
        nt, nv, nte = _causal_counts_per_recording(n)
        train_masks.append(order[:nt])
        valid_masks.append(order[nt : nt + nv])
        test_masks.append(order[nt + nv : nt + nv + nte])

    train_idx = (
        np.concatenate(train_masks)
        if len(train_masks) > 0
        else np.array([], dtype=int)
    )
    valid_idx = (
        np.concatenate(valid_masks)
        if len(valid_masks) > 0
        else np.array([], dtype=int)
    )
    test_idx = (
        np.concatenate(test_masks)
        if len(test_masks) > 0
        else np.array([], dtype=int)
    )

    train_iv = intervals.select_by_mask(
        np.isin(np.arange(len(intervals)), train_idx)
    )
    valid_iv = intervals.select_by_mask(
        np.isin(np.arange(len(intervals)), valid_idx)
    )
    test_iv = intervals.select_by_mask(
        np.isin(np.arange(len(intervals)), test_idx)
    )
    train_iv.sort()
    valid_iv.sort()
    test_iv.sort()
    return train_iv, valid_iv, test_iv
