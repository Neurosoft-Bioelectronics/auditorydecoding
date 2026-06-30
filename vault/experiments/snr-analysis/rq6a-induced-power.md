---
id: rq6a
title: "Induced Power Decomposition"
experiment: "[[snr-analysis/_overview]]"
status: completed
outcome: strongly-supported
created: 2025-06-15
updated: 2025-06-30
tags:
  - signal-quality
  - snr
  - induced-power
  - phase-locking
depends_on:
  - "[[rq6-power-ratio-snr]]"
leads_to: []
---

# RQ6 Addendum: Induced Power Decomposition

## Question

Is the high power ratio driven by induced (non-phase-locked) power rather than evoked power? Does induced power dominate total single-trial power?

## Hypothesis

If neural responses are present but not phase-locked (Hypothesis A from [[rq6-power-ratio-snr]]), then induced power should dominate total single-trial power, and Induced SNR should strongly correlate with Power Ratio SNR but not with Evoked SNR.

## Background

Total single-trial power can be decomposed into two orthogonal components:

- **Evoked power:** Var(ERP) — the variance of the trial-averaged waveform (phase-locked)
- **Induced power:** mean[Var(S)] − Var(ERP) — residual single-trial variance after removing phase-locked component

The additive relationship: Power Ratio SNR = Evoked SNR + Induced SNR.

## Method

$$\text{Induced\_var}_c = \frac{1}{N}\sum_{i=1}^{N}\text{Var}(S_{i,c}(t)) - \text{Var}(\text{ERP}_c(t))$$

$$\text{Induced\_SNR}_c = \frac{\text{Induced\_var}_c}{\frac{1}{N}\sum_{i=1}^{N}\text{Var}(R_{i,c}(t))}$$

- Script: `scripts/run_snr_analysis.py`
- Module: `auditorydecoding/analysis/snr.py`

## Results

| Metric | Value |
|--------|-------|
| Mean Induced SNR (all) | 0.909 |
| Median Induced SNR | 0.957 |
| Mean Evoked SNR (all) | 0.064 |
| Mean Power Ratio SNR (all) | 0.973 |
| Channels with Induced > Evoked | 735 / 736 (99.9%) |

**Correlations:**

| Pair | Spearman ρ | p-value |
|------|-----------|---------|
| Induced SNR vs. Power Ratio SNR | 0.832 | 2.87 × 10⁻¹⁹⁰ |
| Induced SNR vs. Evoked SNR | −0.357 | 1.64 × 10⁻²³ |

Induced power accounts for **93.4%** of total single-trial stimulus variance.

## Interpretation

Induced power overwhelmingly dominates. 99.9% of channels have Induced SNR > Evoked SNR. The strong correlation with Power Ratio SNR (ρ = 0.832) confirms most of the power ratio comes from non-phase-locked activity. The weak anti-correlation with Evoked SNR (ρ = −0.357) suggests channels with the strongest neural responses tend to respond with variable timing, making them invisible to the ERP-based metric. This directly confirms Hypothesis A: low evoked SNR is due to phase-locking failure, not absence of neural responses.

## Verdict

**Outcome:** Strongly Supported — Induced power accounts for 93.4% of total single-trial variance. 99.9% of channels have Induced SNR > Evoked SNR.

## Artifacts

- [Full HTML Report](results/snr_analysis_report.html)

## Links

- Parent: [[snr-analysis/_overview]]
- Depends on: [[rq6-power-ratio-snr]]
- Related: [[rq7-habituation]]
