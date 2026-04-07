# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "mne-bids==0.18",
#   "scikit-learn==1.7.2",
#   "brainsets",
#   "../../../auditorydecoding",
# ]
# ///

from auditorydecoding import NeurosoftPipeline


class Pipeline(NeurosoftPipeline):
    brainset_id = "neurosoft_minipigs_2026"

    # unannotated sessions
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
