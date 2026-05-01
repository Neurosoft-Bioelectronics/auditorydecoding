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
    brainset_id = "neurosoft_monkeys_2026"

    split_config = {
        "test_subjects": {"sub-02"},
        "test_subject_early_sessions": {
            "sub-02": {"ses-01", "ses-02"},
        },
        "folds": [
            {
                "intersubject_valid_subjects": {"sub-04"},
                "intersession_valid_sessions": {
                    ("sub-01", "ses-13"),
                    ("sub-01", "ses-14"),
                    ("sub-01", "ses-15"),
                    ("sub-01", "ses-16"),
                },
            },
            {
                "intersubject_valid_subjects": {"sub-06"},
                "intersession_valid_sessions": {
                    ("sub-01", "ses-13"),
                    ("sub-01", "ses-14"),
                    ("sub-01", "ses-15"),
                    ("sub-01", "ses-16"),
                },
            },
        ],
    }

    skip_sessions = []
