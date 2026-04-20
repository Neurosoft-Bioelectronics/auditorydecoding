# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "mne-bids==0.18",
#   "scikit-learn==1.7.2",
#   "brainsets",
#   "auditorydecoding@git+https://github.com/Neurosoft-Bioelectronics/auditorydecoding@suarez/main",
# ]
# ///

from auditorydecoding import NeurosoftPipeline


class Pipeline(NeurosoftPipeline):
    brainset_id = "neurosoft_monkeys_2026"

    # unannotated sessions
    skip_sessions = []
