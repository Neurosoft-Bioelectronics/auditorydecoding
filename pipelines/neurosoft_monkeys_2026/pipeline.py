# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "mne-bids==0.18",
#   "scikit-learn==1.7.2",
#   "torch-brain",
# ]
# ///


from auditorydecoding import NeurosoftPipeline


class Pipeline(NeurosoftPipeline):
    brainset_id = "nsb-monkeys-v1.1"

    split_config = {
        "test_subjects": {"sub-02"},
        "test_subject_early_sessions": {
            "sub-02": {"ses-01", "ses-02"},
        },
        "intersubject_subjects": [
            "sub-01",
            "sub-03",
            "sub-04",
            "sub-05",
            "sub-06",
        ],
        "intersession_sessions": {
            "sub-01": [
                "ses-01",
                "ses-02",
                "ses-03",
                "ses-04",
                "ses-05",
                "ses-06",
                "ses-07",
                "ses-08",
                "ses-09",
                "ses-10",
                "ses-11",
                "ses-12",
                "ses-13",
                "ses-14",
                "ses-15",
                "ses-16",
            ],
            "sub-03": ["ses-01"],
            "sub-04": ["ses-01"],
            "sub-05": ["ses-01"],
            "sub-06": ["ses-01"],
        },
        "intersession_train_ratio": 0.7,
    }

    skip_sessions = []
