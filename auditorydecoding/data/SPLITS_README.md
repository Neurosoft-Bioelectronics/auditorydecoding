# Neurosoft Split Architecture

This document describes the manual parallel-split design used by the Neurosoft
pipelines for cross-subject and cross-session evaluation.

## Overview

Each processed session file (`.h5`) stores two assignment strings per fold that
control how that session participates in training and evaluation:

- **`intersubject_fold_{k}_assignment`** — for evaluating generalisation to
  *unseen subjects*.
- **`intersession_fold_{k}_assignment`** — for evaluating generalisation to
  *unseen sessions* from subjects seen during training.

There are **2 folds** (`k = 0, 1`). The test vault is constant across folds;
only the validation/training boundary rotates. This gives cross-validated
validation estimates by using a different subject for intersubject validation
in each fold.

### Assignment values

| Value | Meaning |
|-------|---------|
| `"train"` | Session participates in training |
| `"valid"` | Session is used for validation during training |
| `"test"` | Session is reserved for final held-out evaluation |
| `"excluded"` | Session is invisible to this split type (returns empty intervals for every query) |


## The Bucket Design

Sessions are placed into buckets. Each bucket maps to a different assignment
depending on which split type is active:

| Bucket | Purpose | `intersubject` | `intersession` |
|--------|---------|---------------|----------------|
| **Test Vault (early sessions)** | Baseline sessions of test subjects | `"test"` | `"train"` |
| **Test Vault (late sessions)** | Later sessions of test subjects for temporal-drift testing | `"test"` | `"test"` |
| **Intersubject Valid** | Entire subject for cross-subject validation during training | `"valid"` | `"train"` |
| **Intersession Valid** | Latest sessions from training subjects for temporal-drift validation | `"train"` | `"valid"` |
| **Core Training** | Everything else | `"train"` | `"train"` |

The key difference for test-vault subjects: in the **intersubject** split all
their sessions are `"test"` (the model never sees them). In the
**intersession** split their early sessions go into `"train"` so the model
learns each subject's baseline, then later sessions are `"test"` for
temporal-drift evaluation.

Which subjects/sessions populate the Intersubject Valid and Intersession Valid
buckets differs per fold (see per-dataset tables below). The test vault and
the bucket logic are constant.


## Dataset 1 — Minipigs (`neurosoft_minipigs_2026`)

**Subjects:** 01, 02, 03, 04, 05, 06, 07

**Test Vault (both folds):** sub-04 (4 sessions) and sub-07 (5 sessions)

- `intersubject`: all sessions are `"test"`
- `intersession`: ses-01, ses-02 are `"train"` (early baseline); remaining
  sessions are `"test"` (temporal-drift evaluation)

### Fold 0

| Bucket | Sessions |
|--------|----------|
| Intersubject Valid | All of **sub-02** (2 sessions) |
| Intersession Valid | `sub-01_ses-02`, `sub-03_ses-07`, `sub-05_ses-02` |
| Core Training | `sub-01_ses-01`, `sub-03_ses-{01,03,04,06}`, `sub-05_ses-01`, `sub-06_ses-02` |

### Fold 1

Sub-05 rotates into intersubject validation; sub-02 moves to training (its
last session enters intersession validation).

| Bucket | Sessions |
|--------|----------|
| Intersubject Valid | All of **sub-05** (2 sessions) |
| Intersession Valid | `sub-01_ses-02`, `sub-02_ses-02`, `sub-03_ses-07` |
| Core Training | `sub-01_ses-01`, `sub-02_ses-01`, `sub-03_ses-{01,03,04,06}`, `sub-06_ses-02` |

### Minipigs-specific rules

- **Anesthesia sessions** (`acq-LHanest`, `acq-RHanest` on sub-03): forced to
  `"train"` for both split types — never used for evaluation. Brain activity
  under anaesthesia is qualitatively different and would corrupt validation
  metrics.
- **`desc-filtered` sessions**: assigned `"excluded"` for both split types.
  They are still processed and stored (available for intrasession analysis) but
  do not participate in cross-subject or cross-session evaluation.
- **Hemisphere (`acq-LH` / `acq-RH`)**: both hemispheres of the same
  subject+session receive the same assignment. They are treated as separate
  recording files but belong to the same bucket.


## Dataset 2 — Monkeys (`neurosoft_monkeys_2026`)

**Subjects:** 01, 02, 03, 04, 05, 06

**Test Vault (both folds):** sub-02 (5 sessions)

- `intersubject`: all sessions are `"test"`
- `intersession`: ses-01, ses-02 are `"train"` (early baseline); ses-03,
  ses-04, ses-05 are `"test"` (temporal-drift evaluation)

The intersession validation is identical across folds because sub-01 is the
only multi-session training subject regardless of which single-session subject
rotates into intersubject validation.

### Fold 0

| Bucket | Sessions |
|--------|----------|
| Intersubject Valid | **sub-04** (1 session) |
| Intersession Valid | `sub-01_ses-{13,14,15,16}` |
| Core Training | `sub-01_ses-{01..12}`, `sub-03_ses-01`, `sub-05_ses-01`, `sub-06_ses-01` |

### Fold 1

Sub-06 rotates into intersubject validation; sub-04 moves to training.

| Bucket | Sessions |
|--------|----------|
| Intersubject Valid | **sub-06** (1 session) |
| Intersession Valid | `sub-01_ses-{13,14,15,16}` |
| Core Training | `sub-01_ses-{01..12}`, `sub-03_ses-01`, `sub-04_ses-01`, `sub-05_ses-01` |

### Subject imbalance warning

The monkey training set is heavily skewed: sub-01 contributes 12 sessions while
the other training subjects contribute 1 session each. Without correction the
model will overfit to sub-01's baseline activity.

Use `NeurosoftDataset.get_subject_sampling_weights(split="train")` to obtain
per-recording weights that equalise total weight across subjects. Feed these
into a `WeightedRandomSampler` or scale the loss accordingly:

```python
dataset = NeurosoftMonkeys2026(
    root="data/processed",
    split_type="intersession",
    fold_num=0,
    task_type="on_vs_off",
)
weights = dataset.get_subject_sampling_weights(split="train")
# weights[rid] = 1/(sessions_for_subject * num_subjects)
# sub-01 sessions: 1/(12*4) = 0.0208 each
# sub-03/05/06:    1/(1*4)  = 0.25   each
```


## How to Run Each Training Scenario

There are two `split_type` modes for cross-subject/session evaluation. Each
uses the same four buckets but differs in what goes into train vs. valid.
Run each fold separately and average the validation metrics for a
cross-validated estimate.

### `"intersubject"` — evaluate cross-subject generalisation

Train on all `"train"` sessions, validate on the held-out subject, test on the
vault subjects.

```python
for fold in (0, 1):
    dataset = NeurosoftMinipigs2026(
        root="data/processed",
        split_type="intersubject",
        fold_num=fold,
        task_type="on_vs_off",
    )
    train_intervals = dataset.get_sampling_intervals("train")
    valid_intervals = dataset.get_sampling_intervals("valid")
    test_intervals  = dataset.get_sampling_intervals("test")
    # fold 0: valid = sub-02, fold 1: valid = sub-05
```

The intersubject training set includes the intersession-validation sessions
(e.g. `sub-01_ses-02`) because from a subject perspective those subjects are
in the training pool.

### `"intersession"` — evaluate temporal-drift generalisation

Train on earlier sessions, validate on later sessions from the same subjects.

```python
for fold in (0, 1):
    dataset = NeurosoftMinipigs2026(
        root="data/processed",
        split_type="intersession",
        fold_num=fold,
        task_type="on_vs_off",
    )
    train_intervals = dataset.get_sampling_intervals("train")
    valid_intervals = dataset.get_sampling_intervals("valid")
    test_intervals  = dataset.get_sampling_intervals("test")
```

The intersession training set includes the intersubject-validation subject
(e.g. sub-02 in fold 0) and the early sessions from test-vault subjects
(e.g. sub-04 ses-01/02) to maximise training data.

**Important:** the intersubject and intersession splits are designed to be used
in **separate training runs**. Each mode's training set includes sessions that
are validation for the other mode, so monitoring both validation metrics
simultaneously would involve data leakage.


## Other split types (unchanged)

The intrasession splits are orthogonal to the cross-subject/session design and
are unaffected by this architecture:

- **`intrasession-block`** — stratified random train/valid/test within each
  session file (3 folds).
- **`intrasession-causal`** — chronological 60/10/30 train/valid/test within
  each recording (single partition).

These splits operate on trials *within* a session and do not depend on the
intersubject/intersession assignment.


## Configuration reference

Split configs are defined as module-level constants in
`auditorydecoding/data/neurosoft_pipeline.py`:

- `MINIPIGS_SPLIT_CONFIG`
- `MONKEYS_SPLIT_CONFIG`
- `SPLIT_CONFIGS` (maps `brainset_id` to the appropriate config)

Each config has:
- `test_subjects` — set of subjects in the test vault (constant across folds)
- `test_subject_early_sessions` — dict mapping each test subject to its early
  sessions that go into intersession training
- `folds` — list of per-fold dicts, each with `intersubject_valid_subjects`
  and `intersession_valid_sessions`

To add more folds, append another entry to the `folds` list and re-run the
processing pipeline.
