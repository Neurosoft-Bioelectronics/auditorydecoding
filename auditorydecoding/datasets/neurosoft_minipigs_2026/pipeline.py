# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "mne-bids==0.18",
#   "scikit-learn==1.7.2",
#   "temporaldata @ git+https://github.com/neuro-galaxy/temporaldata@main",
# ]
# ///

from argparse import ArgumentParser, Namespace

from typing import Optional

import warnings
import h5py

import numpy as np
import pandas as pd
from datetime import datetime, timezone
import mne
from mne_bids import read_raw_bids, get_entities_from_fname, BIDSPath

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
    "stim_800Hz": 5,
    "stim_1000Hz": 6,
    "stim_2000Hz": 7,
    "stim_5000Hz": 8,
    "stim_8000Hz": 9,
    "stim_10000Hz": 10,
    "stim_15000Hz": 11,
    "stim_16000Hz": 12,
    "stim_20000Hz": 13,
    "stim_30000Hz": 14,
    "stim_40000Hz": 15,
}


# TODO All recordings within these sessions are unnnotated.
# The ones commented out only have a few unannotated recordings.
SKIP_UNANNOTATED_SESSIONS = [
    "sub-03_ses-02_task-AcousStim_acq-RH_desc-raw",
    "sub-03_ses-03_task-AcousStim_acq-RH_desc-raw",
    "sub-03_ses-04_task-AcousStim_acq-RH_desc-raw",
    "sub-03_ses-05_task-AcousStim_acq-RH_desc-raw",
    # "sub-04_ses-02_task-AcousStim_acq-LH_desc-raw", # Some recordings are not annotated
    # "sub-04_ses-02_task-AcousStim_acq-RH_desc-raw", # Some recordings are not annotated
    "sub-05_ses-03_task-AcousStim_acq-LH_desc-raw",
    "sub-05_ses-03_task-AcousStim_acq-RH_desc-raw",
    "sub-06_ses-01_task-AcousStim_acq-LH_desc-raw",
    "sub-06_ses-01_task-AcousStim_acq-RH_desc-raw",
    "sub-07_ses-06_task-AcousStim_acq-LH_desc-filtered",
    "sub-07_ses-06_task-AcousStim_acq-LH_desc-raw",
]


class Pipeline(BrainsetPipeline):
    brainset_id = "neurosoft_minipigs_2026"
    modality = "ieeg"
    parser = parser

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
        session_id = download_output.get("session_id")
        entities = get_entities_from_fname(session_id, on_error="raise")
        subject_id = f"sub-{entities['subject']}"

        if session_id in SKIP_UNANNOTATED_SESSIONS:
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
            "This dataset contains electrophysiology data from minipigs undergoing acoustic stimulation at various frequencies. "
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
            bids_root=self.raw_dir, subject_id=subject_id
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

        self.update_status(f"Extracting {self.modality.upper()} Signal")
        signal = extract_signal(recordings)

        self.update_status("Building Channels")
        channels = extract_channels(raw)

        self.update_status("Extracting behavior intervals")
        on_vs_off_trials = extract_on_vs_off_trials(recordings)
        acoustic_stim_trials = extract_acoustic_stim_trials(recordings)

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
            on_vs_off_splits[f"on_vs_off_fold_{k}_train"] = fold_data.train
            on_vs_off_splits[f"on_vs_off_fold_{k}_valid"] = fold_data.valid
            on_vs_off_splits[f"on_vs_off_fold_{k}_test"] = fold_data.test

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
            acoustic_stim_splits[f"acoustic_stim_fold_{k}_train"] = (
                fold_data.train
            )
            acoustic_stim_splits[f"acoustic_stim_fold_{k}_valid"] = (
                fold_data.valid
            )
            acoustic_stim_splits[f"acoustic_stim_fold_{k}_test"] = (
                fold_data.test
            )

    return Data(
        **namespaced_assignments,
        **on_vs_off_splits,
        **acoustic_stim_splits,
        domain=on_vs_off_trials | acoustic_stim_trials,
    )


def extract_on_vs_off_trials(recordings: dict[str, mne.io.Raw]) -> Interval:
    """
    Extracts 'on' (stimulation) and 'off' (rest and baseline) trials from a collection of Raw objects.

    Args:
        recordings (dict[str, mne.io.Raw]):
            A dictionary mapping recording names to MNE Raw objects from which on/off trial intervals
            will be extracted.

    Returns:
        Interval:
            An Interval object containing all detected on and off (stimulation, rest, and baseline) trials.
    """
    start_times = []
    end_times = []
    labels = []
    first_meas_date = None
    for recording_id, raw in recordings.items():
        meas_date = raw.info["meas_date"]

        if first_meas_date is None:
            first_meas_date = meas_date

        offset = (meas_date - first_meas_date).total_seconds()

        # check if the offset is greater than 1 hour
        if offset > 3600:
            warnings.warn(
                f"Recording {recording_id} with meas_date {meas_date} has offset {offset:.2f} seconds ({offset / 60:.2f} minutes) (> 1 hour) from first_meas_date {first_meas_date}."
            )

        for annotation in raw.annotations:
            start_time = annotation["onset"] + offset
            end_time = start_time + annotation["duration"]

            if len(end_times) > 0 and start_time < end_times[-1]:
                # the previous trial goes into the next one
                # this happens because of numerical precision issues
                assert end_times[-1] - start_time < 0.1, (
                    f"found overlap between trials: start time of trial i: {start_time}, end time of trial i-1: {end_times[-1]}"
                )

                # we can clip the end time of the last trial
                end_times[-1] = start_time

            # skip white-noise stimulation trials
            if (
                "stim" in annotation["description"]
                and "white-noise" in annotation["description"]
            ):
                continue

            start_times.append(start_time)
            end_times.append(end_time)

            if (
                annotation["description"] == "rest"
                or annotation["description"] == "baseline"
            ):
                labels.append("off")
            elif (
                "stim" in annotation["description"]
                and "Hz" in annotation["description"]
            ):
                labels.append("on")

    # Map each unique label to an integer (in increasing order)
    label_ids = [ON_VS_OFF_TO_ID[label] for label in labels]

    return Interval(
        start=np.array(start_times),
        end=np.array(end_times),
        timestamps=(np.array(start_times) + np.array(end_times)) / 2,
        behavior_labels=np.array(labels),
        behavior_ids=np.array(label_ids),
        timekeys=["start", "end", "timestamps"],
    )


def extract_acoustic_stim_trials(recordings: dict[str, mne.io.Raw]) -> Interval:
    """Extracts the acoustic stimulation trials across multiple raw recordings.

    Iterates over all provided raw recordings, extracts annotations corresponding to stimulation
    trials (those with "stim" and "Hz" in their description), and collects their onset and duration
    as trial intervals. Stimulus frequency is extracted and included as part of the trial label.

    Args:
        recordings (dict[str, mne.io.Raw]):
            Dictionary mapping names/keys to raw MNE objects to extract stimulation trials from.

    Returns:
        Interval:
            An Interval object containing start, end, and label information for each detected
            acoustic stimulation trial, along with label IDs representing the stimulation frequency.
    """
    start_times = []
    end_times = []
    labels = []
    first_meas_date = None
    for recording_id, raw in recordings.items():
        meas_date = raw.info["meas_date"]

        if first_meas_date is None:
            first_meas_date = meas_date

        offset = (meas_date - first_meas_date).total_seconds()

        # check if the offset is greater than 1 hour
        if offset > 3600:
            warnings.warn(
                f"Recording {recording_id} with meas_date {meas_date} has offset {offset:.2f} seconds ({offset / 60:.2f} minutes) (> 1 hour) from first_meas_date {first_meas_date}."
            )

        for annotation in raw.annotations:
            start_time = annotation["onset"] + offset
            end_time = start_time + annotation["duration"]

            if len(end_times) > 0 and start_time < end_times[-1]:
                # the previous trial goes into the next one
                # this happens because of numerical precision issues
                assert end_times[-1] - start_time < 0.1, (
                    f"found overlap between trials: start time of trial i: {start_time}, end time of trial i-1: {end_times[-1]}"
                )

                # we can clip the end time of the last trial
                end_times[-1] = start_time

            if (
                "stim" in annotation["description"]
                and "Hz" in annotation["description"]
            ):
                start_times.append(start_time)
                end_times.append(end_time)

                # extract the stimulation frequency
                frequency = _extract_stim_frequency(annotation["description"])
                labels.append(f"stim_{frequency}Hz")

    # Map each unique label to an integer (in increasing order)
    label_ids = [STIM_FREQUENCY_TO_ID[label] for label in labels]

    return Interval(
        start=np.array(start_times),
        end=np.array(end_times),
        timestamps=(np.array(start_times) + np.array(end_times)) / 2,
        behavior_labels=np.array(labels),
        behavior_ids=np.array(label_ids),
        timekeys=["start", "end", "timestamps"],
    )


def extract_signal(recordings: dict[str, mne.io.Raw]) -> IrregularTimeSeries:
    """Extracts the signal from a session of recordings.

    Timestamps are computed relative to the start of the first recording (0),
    with each subsequent recording offset by the actual wall-clock elapsed time
    since the first recording started (via meas_date). This preserves gaps
    between recordings in the timeline.

    Args:
        session (dict[str, mne.io.Raw]): The session of recordings to extract the signal from.
            Expected to be sorted by meas_date (via _sort_recordings).

    Returns:
        IrregularTimeSeries: The extracted signal with relative timestamps.
    """
    signal = []
    timestamps = []
    first_meas_date = None

    for _, raw in recordings.items():
        signal.append(raw.get_data())
        meas_date = raw.info["meas_date"]

        if first_meas_date is None:
            first_meas_date = meas_date

        offset = (meas_date - first_meas_date).total_seconds()
        ts = raw.times.astype(np.float64) + offset
        timestamps.append(ts[:, np.newaxis])

    return IrregularTimeSeries(
        signal=np.hstack(signal).T,
        timestamps=np.vstack(timestamps).squeeze(),
        domain="auto",
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
        dict[str, mne.io.Raw]: The dictionary of loaded recordings.
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

    new_on_vs_off_trials = Interval(
        start=start_times,
        end=end_times,
        behavior_labels=behavior_labels,
        behavior_ids=behavior_ids,
        timestamps=(start_times + end_times) / 2,
        timekeys=["start", "end", "timestamps"],
    )
    new_on_vs_off_trials.sort()

    return new_on_vs_off_trials
