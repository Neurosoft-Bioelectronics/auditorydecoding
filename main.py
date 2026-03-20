import time
from auditorydecoding.datasets.neurosoft_minipigs_2026.NeurosoftMinipgs2026 import (
    NeurosoftMinipigs2026,
)
from auditorydecoding.features import FFTFeatures
from auditorydecoding.windowing import extract_windows

dataset = NeurosoftMinipigs2026(
    root="data/processed",
    recording_ids=["sub-02_ses-01_task-AcousticStim_desc-raw_LH"],
    fold_num=0,
    split_type="intrasession",
    task_type="on_vs_off",
)

t0 = time.time()
X_train, y_train = extract_windows(
    dataset, "train", 0.5, feature_extractor=FFTFeatures()
)
t1 = time.time()
print(f"extract_windows (train): {t1 - t0:.1f}s  ->  X={X_train.shape}")

X_test, y_test = extract_windows(
    dataset, "test", 0.5, feature_extractor=FFTFeatures()
)
t2 = time.time()
print(f"extract_windows (test):  {t2 - t1:.1f}s  ->  X={X_test.shape}")
print(f"total: {t2 - t0:.1f}s")
