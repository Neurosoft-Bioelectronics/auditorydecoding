# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "mne-bids==0.18",
#   "scikit-learn==1.7.2",
#   "brainsets",
#   "auditorydecoding@git+https://github.com/Neurosoft-Bioelectronics/auditorydecoding@main",
# ]
# ///

from auditorydecoding import NeurosoftPipeline


class Pipeline(NeurosoftPipeline):
    brainset_id = "neurosoft_minipigs_2026"

    split_config = {
        "test_subjects": {"sub-04", "sub-07"},
        "test_subject_early_sessions": {
            "sub-04": {"ses-01", "ses-02"},
            "sub-07": {"ses-01", "ses-02"},
        },
        "intersubject_subjects": [
            "sub-01",
            "sub-02",
            "sub-03",
            "sub-05",
            "sub-06",
        ],
        "intersession_sessions": {
            "sub-01": ["ses-01", "ses-02"],
            "sub-02": ["ses-01", "ses-02"],
            "sub-03": ["ses-01", "ses-03", "ses-04", "ses-06", "ses-07"],
            "sub-05": ["ses-01", "ses-02"],
            "sub-06": ["ses-02"],
        },
        "intersession_train_ratio": 0.7,
    }

    skip_sessions = [
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
