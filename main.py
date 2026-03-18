import time
from auditorydecoding.datasets.neurosoft_minipigs_2026.NeurosoftMinipgs2026 import (
    NeurosoftMinipigs2026,
)
from auditorydecoding.preprocessing import (
    PreprocessingPipeline,
    ChannelSelector,
    ZeroCenter,
)
from auditorydecoding.features import FFTFeatures
from auditorydecoding.windowing import extract_windows

dataset = NeurosoftMinipigs2026(
    root="data/processed",
    recording_ids=["sub-02_ses-01_task-AcousticStim_desc-raw_LH"],
    fold_num=0,
    split_type="intrasession",
    task_type="on_vs_off",
    preprocessing=PreprocessingPipeline(
        [
            ChannelSelector(channel_types=["ecog"]),
            ZeroCenter(),
        ]
    ),
)

t0 = time.time()
dataset.preprocess(split="train")
t1 = time.time()
print(f"preprocess: {t1 - t0:.1f}s")

t2 = time.time()
X_train, y_train = extract_windows(
    dataset, "train", 0.5, feature_extractor=FFTFeatures()
)
t3 = time.time()
print(f"extract_windows (train): {t3 - t2:.1f}s  ->  X={X_train.shape}")

X_test, y_test = extract_windows(
    dataset, "test", 0.5, feature_extractor=FFTFeatures()
)
t4 = time.time()
print(f"extract_windows (test):  {t4 - t3:.1f}s  ->  X={X_test.shape}")
print(f"total: {t4 - t0:.1f}s")
