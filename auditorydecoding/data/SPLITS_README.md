# Neurosoft Data Split Schemas

This document describes the split design used by the Neurosoft pipelines for
cross-subject and cross-session evaluation.

## Overview

Each processed session file (`.h5`) stores assignment strings that control how
that session participates in training and evaluation:

- `**intersubject_fold_{k}_assignment**` — for evaluating generalisation to
*unseen subjects* (leave-one-out cross-validation).
- `**intersession_fold_0_assignment**` — for evaluating generalisation to
*unseen sessions* from subjects seen during training.

The two split types are **fully independent** and designed for **separate
training runs**. Neither references the other's validation set.

### Assignment values


| Value        | Meaning                                                                           |
| ------------ | --------------------------------------------------------------------------------- |
| `"train"`    | Session participates in training                                                  |
| `"valid"`    | Session is used for validation during training                                    |
| `"test"`     | Session is reserved for final held-out evaluation                                 |
| `"excluded"` | Session is invisible to this split type (returns empty intervals for every query) |


## Intersubject Split — Leave-One-Out (LOO)

The intersubject split evaluates how well the model generalises to a **completely
unseen subject**. It uses leave-one-out cross-validation: there is one fold per
subject, and in each fold that subject is held out entirely for
validation while the remaining subjects form the training set.

Furthermore, there is a test set (fixed set of subjects) that is always held out and 
hence does not make part of the LOO rotation. The test set is common across the intersession 
and intersubject splits.

### Assignment rules

| Group                    | Assignment   |
| ------------------------ | ------------ |
| Test subjects            | `"test"`     |
| LOO subject (fold k)     | `"valid"`    |
| All other subjects       | `"train"`    |

The number of folds equals the number of non-test subjects.

## Intersession Split — Causal 70/30

The intersession split evaluates how well the model handles **temporal drift**
— performing on later sessions from subjects it has already seen. It is a
single deterministic fold (no cross-validation).

For each subject with multiple sessions, the sessions are ordered
chronologically: the first ~70% are assigned to training and the remaining
~30% to validation. Subjects with only one session contribute that session to
training only.

For test subjects, early sessions (defined per dataset) go into training so
the model learns each subject's baseline, and the remaining later sessions
are held out as test data.

### Assignment rules

| Group                                    | Assignment |
| ---------------------------------------- | ---------- |
| Test subjects — early sessions           | `"train"`  |
| Test subjects — later sessions           | `"test"`   |
| Non-test subjects — first ~70% sessions  | `"train"`  |
| Non-test subjects — remaining sessions   | `"valid"`  |
| Non-test subjects — single session only  | `"train"`  |

### Train/validation boundary

The number of training sessions for a subject with *N* sessions is
`max(floor(0.7 * N), 1)`. Sessions are taken in chronological order.

## Dataset-specific rules

### Minipigs-specific rules

- **Anesthesia sessions** (`acq-LHanest`, `acq-RHanest` on sub-03): forced to
`"train"` for both split types — never used for evaluation. Brain activity
under anaesthesia is qualitatively different and would corrupt validation
metrics.
- `**desc-filtered` sessions**: assigned `"excluded"` for both split types.
They are still processed and stored (available for intrasession analysis) but
do not participate in cross-subject or cross-session evaluation.
- **Hemisphere (`acq-LH` / `acq-RH`)**: both hemispheres of the same
subject+session receive the same assignment. They are treated as separate
recording files but belong to the same group.

## Dataset 1 — Minipigs (`neurosoft_minipigs_2026`)

**Subjects:** 01, 02, 03, 04, 05, 06, 07

**Test Set:** sub-04 (4 sessions) and sub-07 (5 sessions)

- `intersubject`: all sessions are `"test"`
- `intersession`: ses-01, ses-02 are `"train"` (early baseline); remaining
sessions are `"test"` (temporal-drift evaluation)

### Intersubject — 5 LOO folds

Non-test subjects: sub-01, sub-02, sub-03, sub-05, sub-06


| Fold | Validation subject | Training subjects          |
| ---- | ------------------ | -------------------------- |
| 0    | sub-01             | sub-02, 03, 05, 06         |
| 1    | sub-02             | sub-01, 03, 05, 06         |
| 2    | sub-03             | sub-01, 02, 05, 06         |
| 3    | sub-05             | sub-01, 02, 03, 06         |
| 4    | sub-06             | sub-01, 02, 03, 05         |


sub-04 and sub-07 are `"test"` in every fold.

### Intersession — single fold

Non-test subjects (70/30 chronological):


| Subject | Sessions                      | Training               | Validation     |
| ------- | ----------------------------- | ---------------------- | -------------- |
| sub-01  | ses-01, ses-02                | ses-01                 | ses-02         |
| sub-02  | ses-01, ses-02                | ses-01                 | ses-02         |
| sub-03  | ses-01, 03, 04, 06, 07 (5)   | ses-01, 03, 04         | ses-06, 07     |
| sub-05  | ses-01, ses-02                | ses-01                 | ses-02         |
| sub-06  | ses-02 (1)                    | ses-02                 | —              |


Test subjects:


| Subject | Early (train) | Late (test)          |
| ------- | ------------- | -------------------- |
| sub-04  | ses-01, 02    | ses-03, 04           |
| sub-07  | ses-01, 02    | ses-03, 04, 05       |


## Dataset 2 — Monkeys (`neurosoft_monkeys_2026`)

**Subjects:** 01, 02, 03, 04, 05, 06

**Test Set:** sub-02 (5 sessions)

- `intersubject`: all sessions are `"test"`
- `intersession`: ses-01, ses-02 are `"train"` (early baseline); ses-03,
ses-04, ses-05 are `"test"` (temporal-drift evaluation)

### Intersubject — 5 LOO folds

Non-test subjects: sub-01, sub-03, sub-04, sub-05, sub-06


| Fold | Validation subject | Training subjects          |
| ---- | ------------------ | -------------------------- |
| 0    | sub-01             | sub-03, 04, 05, 06         |
| 1    | sub-03             | sub-01, 04, 05, 06         |
| 2    | sub-04             | sub-01, 03, 05, 06         |
| 3    | sub-05             | sub-01, 03, 04, 06         |
| 4    | sub-06             | sub-01, 03, 04, 05         |


sub-02 is `"test"` in every fold.

### Intersession — single fold

Non-test subjects (70/30 chronological):


| Subject | Sessions                | Training               | Validation     |
| ------- | ----------------------- | ---------------------- | -------------- |
| sub-01  | ses-01 .. ses-16 (16)   | ses-01 .. ses-11 (11)  | ses-12 .. 16   |
| sub-03  | ses-01 (1)              | ses-01                 | —              |
| sub-04  | ses-01 (1)              | ses-01                 | —              |
| sub-05  | ses-01 (1)              | ses-01                 | —              |
| sub-06  | ses-01 (1)              | ses-01                 | —              |


Test subjects:


| Subject | Early (train) | Late (test)          |
| ------- | ------------- | -------------------- |
| sub-02  | ses-01, 02    | ses-03, 04, 05       |


### Subject imbalance warning

The monkey training set is heavily skewed: sub-01 contributes many more
sessions than the other training subjects. Without correction the model will
overfit to sub-01's baseline activity.

Use `NeurosoftDataset.get_subject_sampling_weights(split="train")` to obtain
per-recording weights that equalise total weight across subjects. Feed these
into a `WeightedRandomSampler` or scale the loss accordingly:

```python
dataset = NeurosoftMonkeys2026(
    root="data/processed",
    split_type="intersession",
    task_type="on_vs_off",
)
weights = dataset.get_subject_sampling_weights(split="train")
```

## How to Run Each Training Scenario

### `"intersubject"` — evaluate cross-subject generalisation

Train on all `"train"` sessions, validate on the held-out subject, test on the
test-set subjects. Run each LOO fold separately and average metrics.

```python
n_folds = 5  # one per non-test subject
for fold in range(n_folds):
    dataset = NeurosoftMinipigs2026(
        root="data/processed",
        split_type="intersubject",
        fold_num=fold,
        task_type="on_vs_off",
    )
    train_intervals = dataset.get_sampling_intervals("train")
    valid_intervals = dataset.get_sampling_intervals("valid")
    test_intervals = dataset.get_sampling_intervals("test")
```

### `"intersession"` — evaluate temporal-drift generalisation

Train on earlier sessions, validate on later sessions from the same subjects.
There is a single fold; `fold_num` can be omitted.

```python
dataset = NeurosoftMinipigs2026(
    root="data/processed",
    split_type="intersession",
    task_type="on_vs_off",
)
train_intervals = dataset.get_sampling_intervals("train")
valid_intervals = dataset.get_sampling_intervals("valid")
test_intervals = dataset.get_sampling_intervals("test")
```

## Other split types (unchanged)

The intrasession splits are orthogonal to the cross-subject/session design and
are unaffected by the intersubject/intersession design:

- **intrasession-block** — stratified random train/valid/test within each
session file (3 folds).
- **intrasession-causal** — chronological 60/10/30 train/valid/test within
each recording (single partition).

These splits operate on trials *within* a session and do not depend on the
intersubject/intersession assignment.

## Class balance after frequency-band remapping

Individual stimulus frequencies are remapped to 8 perceptual frequency bands
for classification. The mapping groups the raw `stim_*Hz` labels as follows:

| Band          | Stimulus frequencies                           |
| ------------- | ---------------------------------------------- |
| `low_bass`    | 100, 200, 300, 400, 500, 650 Hz                |
| `mid_bass`    | 800 Hz                                         |
| `low_mids`    | 1000, 1500 Hz                                  |
| `midrange`    | 2000, 3000, 4000 Hz                            |
| `high_mids`   | 5000 Hz                                        |
| `low_treble`  | 7700, 8000, 9500 Hz                            |
| `mid_treble`  | 10000 Hz                                       |
| `high_treble` | 12000, 13000, 15000, 16000, 18000, 20000, 30000, 40000 Hz |

After remapping, the class proportions are relatively even across
train/valid/test splits for all split types and folds. The heatmaps below show
the remapped band proportions (%) for every (split type, fold, split)
configuration; uniform would be 12.5% per band.

### Minipigs

![Minipigs — remapped band proportions across all configurations](figures/minipigs_cross_config_proportions.png)

For minipigs, proportions are stable across all split configurations. The
intrasession splits (block and causal) are near-uniform by construction. The
intersubject and intersession splits show slightly more variation (driven by
per-subject stimulation protocols) but remain well-balanced, with most values
within a few percentage points of the 12.5% uniform baseline.

### Monkeys

![Monkeys — remapped band proportions across all configurations](figures/monkeys_cross_config_proportions.png)

For monkeys, the intrasession and intersubject splits show good balance.
However, the **intersession test split** is a notable outlier: the `high_treble`
band dominates at ~63%, with several bands at 0%.

This is an artifact of the underlying data, not a split-design flaw. Of the
5 monkey subjects, 4 have only a single session with a very limited set of
stimulus frequencies, so they can only be assigned to training. The one subject
with many sessions (sub-01) covers a broad range of frequencies, but that
subject's later sessions must go either into the calibration set (train/valid)
or into the test set, they cannot appear in both. The test subject (sub-02)
happens to have sessions that are heavily skewed toward high frequencies.
There is no practical way to fix this without fundamentally altering the
intersession evaluation protocol (e.g. putting sub-01's sessions in test would
leave almost no data for training). For the monkey dataset, the intersession
test results on the `acoustic_stim` task should therefore be interpreted with
this class imbalance in mind. The intersubject and intrasession evaluations
remain the more reliable benchmarks.

## Configuration reference

Split configs are defined as the `split_config` class attribute on each
per-animal pipeline:

- `pipelines/neurosoft_minipigs_2026/pipeline.py` → `Pipeline.split_config`
- `pipelines/neurosoft_monkeys_2026/pipeline.py` → `Pipeline.split_config`

Each config dict has:

- `test_subjects` — set of subjects in the test set (constant across folds)
- `test_subject_early_sessions` — dict mapping each test subject to its early
sessions that go into intersession training
- `intersubject_subjects` — ordered list of non-test subjects; each index
corresponds to one LOO fold
- `intersession_sessions` — dict mapping each non-test subject to its chronologically
ordered list of sessions
- `intersession_train_ratio` — fraction of sessions per subject used for
training (default 0.7)

To add more subjects or sessions, update the config dict and re-run the
processing pipeline.
