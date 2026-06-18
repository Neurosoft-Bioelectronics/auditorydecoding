# Class Balance Analysis: Frequency Band Remapping

## Overview

This report analyzes the class distribution of acoustic stimulation trials in the **minipigs** and **monkeys** datasets, comparing the original per-frequency labels against the proposed **6-band frequency remapping**. White noise (`stim_wn`) is **excluded entirely** from this analysis.

### Proposed Frequency Band Mapping

| Band | Frequencies | # Freqs |
|------|------------|---------|
| sub_bass | 100, 200, 300, 400, 500 Hz | 5 |
| low | 650, 800, 1000, 1500 Hz | 4 |
| low_mid | 2000, 3000, 4000 Hz | 3 |
| mid | 5000, 7700, 8000, 9500 Hz | 4 |
| high_mid | 10000, 12000, 13000, 15000, 16000 Hz | 5 |
| high | 18000, 20000, 30000, 40000 Hz | 4 |

### Key Questions

1. **How unbalanced are the classes before and after the remapping?**
2. **Are these imbalances consistent across splits, folds, and split types?**
   - Balance between (train + valid) and test
   - Balance between train, valid, and test individually

---

## 1. MINIPIGS

**Total trials (excl. stim_wn): 66,791** across 7 subjects, 16 unique frequencies present.

### 1.1 Q1 — How unbalanced are the classes?

The heatmap below shows class proportions within each split (using intrasession-block fold 0 as a clean reference). With 6 classes, uniform would be 16.67% per band.

![Imbalance overview](minipigs/heatmap_imbalance_overview.png)

#### Aggregate Remapped Distribution

| Band | Count | % | Deviation from uniform |
|------|-------|---|----------------------|
| sub_bass | 11,542 | 17.3% | +0.6pp |
| low | 13,522 | 20.2% | +3.6pp |
| low_mid | 8,034 | 12.0% | -4.6pp |
| mid | 13,916 | 20.8% | +4.2pp |
| high_mid | 13,585 | 20.3% | +3.7pp |
| high | 6,192 | 9.3% | -7.4pp |

**Remapped max/min ratio: 2.25x** (mid at 20.8% vs high at 9.3%)

The "high" band is the smallest (9.3%) because its 4 member frequencies (18–40 kHz) each have relatively few trials. The "low_mid" band is also underrepresented (12.0%) as it has only 3 member frequencies.

#### Before vs After Comparison

The heatmap below shows the max/min ratio across all split types and folds, comparing original (left) vs remapped (right):

![Max/min ratio heatmap](minipigs/heatmap_maxmin_ratio.png)

| Metric | Original | Remapped |
|--------|----------|----------|
| # Classes | 16 | 6 |
| Max/min ratio (intrasession) | 11.3x | **2.25x** |
| Max/min ratio (intersubject train) | 8.9–17.0x | **1.62–1.80x** |
| CV (intrasession) | 0.717 | **0.269** |

**The remapping reduces within-split imbalance by ~80%.** Train splits achieve excellent balance (1.6–1.8x). However, the intersubject test set (sub-04 + sub-07) retains a 7.88x ratio because these subjects have very few "high" band trials (3.1%).

### 1.2 Q2 — Are imbalances consistent across splits?

#### All configurations at a glance

![Cross-config proportions](minipigs/heatmap_cross_config_proportions.png)

#### Intrasession splits — Near-perfect consistency

| Split Type | Fold | train max/min | valid max/min | test max/min |
|------------|------|---------------|---------------|--------------|
| intrasession-block | 0 | 2.24 | 2.25 | 2.25 |
| intrasession-block | 1 | 2.25 | 2.25 | 2.24 |
| intrasession-block | 2 | 2.25 | 2.24 | 2.25 |
| intrasession-causal | 0 | 2.25 | 2.23 | 2.25 |

**Verdict:** Proportions are virtually identical across all splits and folds. Train+valid vs test drift is < 0.1pp. This is the ideal baseline.

#### Train+Valid vs Test drift

![TV vs test drift](minipigs/heatmap_trainval_vs_test_drift.png)

- **Intrasession:** < 0.1pp drift on all bands — essentially zero
- **Intersubject (all folds):** 9.83pp drift on "high" band — this is constant across folds because the test set is always sub-04 + sub-07, who have only 3.1% "high" vs ~12–14% in training
- **Intersession:** 11.86pp drift on "high" — the later sessions of test subjects have different frequency coverage

#### Train vs Valid vs Test individual drift

![Train/valid/test drift](minipigs/heatmap_train_valid_test_drift.png)

The intersubject splits show 10–23pp max drift between train/valid/test because each subject has a distinct frequency profile:

| Subject(s) | sub_bass | low | low_mid | mid | high_mid | high | Total |
|------------|----------|-----|---------|-----|----------|------|-------|
| sub-01 (valid f0) | 21.3% | 15.0% | 7.6% | 15.1% | 17.9% | **23.1%** | 3,992 |
| sub-02 (valid f1) | 22.1% | 11.8% | 13.3% | 16.2% | 20.9% | 15.8% | 6,050 |
| sub-03 (valid f2) | 17.8% | 21.4% | 15.9% | 16.8% | 19.8% | 8.4% | 7,123 |
| sub-05 (valid f3) | 22.6% | 19.9% | 12.2% | 24.0% | 21.3% | **0.0%** | 4,162 |
| sub-06 (valid f4) | 13.3% | 20.1% | 13.3% | 13.3% | 26.7% | 13.3% | 3,624 |
| sub-04+07 (test) | 17.2% | 24.0% | 10.6% | 24.0% | 21.1% | **3.1%** | 15,789 |

**Key observation:** sub-05 has **zero "high" trials** and test subjects (sub-04+07) have only 3.1% "high". This is the primary driver of intersubject imbalance.

### 1.3 Intersession Split

| Split | Subjects | Trials | Max/Min | # Classes |
|-------|----------|--------|---------|-----------|
| train | all subjects | 31,627 | 1.88 | 6 |
| valid | sub-03,04,05,07 | 12,265 | 2.92 | 6 |
| test | sub-02,03,04,05,07 | 8,584 | 1.89 | **5** |

The intersession test set is **missing one frequency band** (only 5 of 6 classes present). The 11.86pp drift on "high" suggests test sessions have limited high-frequency coverage.

---

## 2. MONKEYS

**Total trials (excl. stim_wn): 39,606** across 6 subjects, 22 unique frequencies present.

### 2.1 Q1 — How unbalanced are the classes?

![Imbalance overview](monkeys/heatmap_imbalance_overview.png)

#### Aggregate Remapped Distribution

| Band | Count | % | Deviation from uniform |
|------|-------|---|----------------------|
| sub_bass | 5,883 | 14.9% | -1.8pp |
| low | 8,399 | 21.2% | +4.5pp |
| low_mid | 5,584 | 14.1% | -2.6pp |
| mid | 8,811 | 22.2% | +5.6pp |
| high_mid | 8,130 | 20.5% | +3.9pp |
| high | 2,799 | 7.1% | **-9.6pp** |

**Remapped max/min ratio: 3.15x** (mid at 22.2% vs high at 7.1%)

The "high" band is the most underrepresented, containing only 7.1% of trials. The original per-frequency distribution was **extremely** unbalanced (184.7x) due to several frequencies having only 30–120 trials.

#### Before vs After Comparison

![Max/min ratio heatmap](monkeys/heatmap_maxmin_ratio.png)

| Metric | Original | Remapped |
|--------|----------|----------|
| # Classes | 22 | 6 |
| Max/min ratio (intrasession) | 184.7x | **3.15x** |
| Max/min ratio (intersubject train) | 16.9–132.6x | **3.0–21.9x** |
| CV (intrasession) | 0.915 | **0.318** |

**The remapping reduces intrasession imbalance by 98%.** However, intersubject splits remain challenging.

### 2.2 Q2 — Are imbalances consistent across splits?

#### All configurations at a glance

![Cross-config proportions](monkeys/heatmap_cross_config_proportions.png)

#### Intrasession splits — Excellent consistency

| Split Type | Fold | train max/min | valid max/min | test max/min |
|------------|------|---------------|---------------|--------------|
| intrasession-block | 0 | 3.14 | 3.13 | 3.17 |
| intrasession-block | 1 | 3.15 | 3.16 | 3.14 |
| intrasession-block | 2 | 3.15 | 3.15 | 3.14 |
| intrasession-causal | 0 | 3.15 | 3.12 | 3.15 |

**Verdict:** Like minipigs, intrasession splits preserve proportions perfectly (< 0.1pp drift). The ~3.15x ratio is purely structural.

#### Train+Valid vs Test drift

![TV vs test drift](monkeys/heatmap_trainval_vs_test_drift.png)

- **Intrasession:** < 0.1pp drift — excellent
- **Intersubject (all folds):** **18.98pp** drift on "high_mid" — the test subject (sub-02) has 34.2% high_mid while train averages ~15%
- **Intersession:** **44.60pp** drift on "high_mid" — the intersession test set (later sessions) has drastically different frequency coverage, with only 3 classes present

#### Train vs Valid vs Test individual drift

![Train/valid/test drift](monkeys/heatmap_train_valid_test_drift.png)

The monkey subjects have **radically different frequency protocols**:

| Subject(s) | sub_bass | low | low_mid | mid | high_mid | high | Total |
|------------|----------|-----|---------|-----|----------|------|-------|
| sub-01 (valid f0) | 18.3% | 23.8% | 13.6% | 19.6% | 15.5% | 9.2% | 19,898 |
| sub-02 (test) | 8.2% | 9.6% | 19.6% | 20.8% | **34.2%** | 7.7% | 11,015 |
| sub-03 (valid f1) | 14.9% | 13.6% | 13.7% | 27.8% | 16.3% | 13.7% | 877 |
| sub-04 (valid f2) | 17.0% | 21.0% | 7.3% | 29.6% | 25.1% | **0.0%** | 4,159 |
| sub-05 (valid f3) | 15.0% | **42.8%** | 9.0% | 30.8% | **2.4%** | **0.0%** | 3,349 |
| sub-06 (valid f4) | **0.0%** | **60.1%** | **0.0%** | 39.9% | **0.0%** | **0.0%** | 308 |

**Critical findings:**
- **sub-06** has only 308 trials and only 2 frequency bands (low + mid)
- **sub-05** has 42.8% "low" and zero "high" trials
- **sub-04** has zero "high" trials
- **sub-02** (always the test subject) has 34.2% "high_mid" — heavily skewed
- Only **sub-01** (19,898 trials) has reasonable coverage of all 6 bands
- sub-01 contributes **50%** of all monkey trials

### 2.3 Intersession Split — Severe issues

| Split | Subjects | Trials | Max/Min | # Classes |
|-------|----------|--------|---------|-----------|
| train | sub-01,02,05 | 27,838 | 3.94 | 6 |
| valid | sub-01,04,06 | 9,562 | 2.69 | 6 |
| test | sub-01,02,03,04 | 2,206 | 13.96 | **3** |

The intersession test set has only **2,206 trials** and **3 classes** out of 6. The 44.60pp drift on high_mid means the test set has a fundamentally different class composition than training.

---

## 3. Answers to Key Questions

### Q1: How unbalanced are the classes before and after the remapping?

| Dataset | Original classes | Original max/min | Remapped classes | Remapped max/min | Improvement |
|---------|-----------------|-----------------|-----------------|-----------------|-------------|
| Minipigs | 16 | 11.3x | 6 | **2.25x** | 80% |
| Monkeys | 22 | 184.7x | 6 | **3.15x** | 98% |

The remapping is highly effective at reducing within-split imbalance. The remaining ~2–3x ratio is structural: bands with more member frequencies (sub_bass: 5, high_mid: 5) naturally accumulate more trials than bands with fewer (low_mid: 3, high: 4). The "high" band is consistently the smallest in both datasets.

### Q2: Are imbalances consistent across splits?

**Intrasession (block and causal):** YES — proportions are near-identical across train/valid/test. Drift < 0.1pp. This is the most reliable evaluation setting.

**Intersubject:** NO — substantial drift driven by per-subject protocol differences.
- Minipigs: 9.83pp max drift (test subjects have 3.1% "high" vs 12%+ in train)
- Monkeys: 18.98pp max drift (test subject has 34.2% "high_mid" vs ~15% in train)

**Intersession:** NO — session-level protocol changes cause severe drift.
- Minipigs: 11.86pp max drift, test missing 1 class
- Monkeys: **44.60pp max drift**, test has only 3 of 6 classes

**Train+Valid vs Test:** The drift is constant across folds for intersubject (since the test set is fixed). The problem is structural: test subjects have different frequency protocols.

**Train vs Valid vs Test:** Individual split proportions vary more, especially in intersubject folds where a single subject serves as validation. Monkeys are particularly affected — sub-06 (fold 4 valid) has only 308 trials across 2 bands.

---

## 4. Recommendations

### 4.1 The remapping works well — use it

The 6-band remapping reduces the classification problem from an extremely unbalanced 16–22 class task to a manageable 6-class task with 2–3x imbalance. This remaining imbalance is easily handled with weighted sampling or loss weighting.

### 4.2 Adjust band boundaries to reduce structural imbalance

The current "high" band is consistently underrepresented (7–9%). Two adjustments would help equalize band sizes:

| Change | Effect |
|--------|--------|
| Move `stim_1500Hz` from "low" → "low_mid" | low: 4→3 freqs, low_mid: 3→4 freqs |
| Move `stim_16000Hz` from "high_mid" → "high" | high_mid: 5→4 freqs, high: 4→5 freqs |

This would give each band 3–5 frequencies with more uniform trial counts.

### 4.3 Use weighted sampling for intersubject/intersession training

Since test-set class proportions differ from training (up to 19pp for monkeys), use:
- **Inverse-frequency class weights** in the loss function
- **Evaluation with balanced accuracy** (or macro-averaged metrics) rather than raw accuracy

### 4.4 Monkey intersubject evaluation requires careful interpretation

The monkey intersubject setting is fundamentally limited:
- sub-02 (test) has a very different frequency profile (34.2% high_mid)
- sub-05, sub-06 lack entire bands (zero "high" trials)
- sub-06 has only 308 trials total

**Recommendations:**
- Report per-class metrics alongside aggregate accuracy
- Consider whether sub-05 and sub-06 should be excluded from intersubject evaluation given their limited protocol
- Note that "high" band performance in intersubject is unreliable since several subjects have no "high" trials at all

### 4.5 Monkey intersession test set is too small and incomplete

With only 2,206 trials and 3 of 6 classes, the intersession test set cannot meaningfully evaluate all bands. Consider:
- Adding more test sessions if available
- Reporting intersession results only for the 3 classes present in test
- Relying on intrasession-causal as the primary temporal-generalization benchmark

### 4.6 Leverage intrasession splits for method development

The intrasession splits produce perfectly balanced and consistent class distributions. Use them for:
- Model development and hyperparameter tuning
- Fair comparison between methods
- Ablation studies

Reserve intersubject/intersession for final generalization claims, with per-fold and per-class results.

---

## Appendix: Generated Plots

All plots are saved in `reports/label_analysis/{dataset}/`:

**Heatmaps (summary across all configurations):**
- `heatmap_imbalance_overview.png` — Q1: original vs remapped proportions
- `heatmap_maxmin_ratio.png` — Q1: max/min ratio before/after across all configs
- `heatmap_cross_config_proportions.png` — Q2: band proportions across all configs
- `heatmap_trainval_vs_test_drift.png` — Q2: (train+valid) vs test drift
- `heatmap_train_valid_test_drift.png` — Q2: train vs valid vs test drift

**Per-configuration bar charts:**
- `{split_type}_{fold}_comparison.png` — side-by-side original vs remapped
- `{split_type}_{fold}_original.png` — original per-frequency distribution
- `{split_type}_{fold}_remapped.png` — remapped band distribution
