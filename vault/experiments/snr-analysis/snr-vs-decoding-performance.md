---
id: snr-decoding
title: "SNR vs. Decoding Performance Across Baselines"
experiment: "[[snr-analysis/_overview]]"
status: completed
outcome: strongly-supported
created: 2025-06-15
updated: 2025-06-30
tags:
  - signal-quality
  - snr
  - decoding
  - correlation
  - baselines
depends_on:
  - "[[rq1-channel-session-snr]]"
  - "[[rq4-tonotopic-tuning]]"
  - "[[rq6-power-ratio-snr]]"
  - "[[rq7-habituation]]"
leads_to:
  - "[[session-consistency]]"
---

# SNR vs. Decoding Performance Across Baselines

## Question

Does SNR predict downstream decoding performance regardless of model architecture?

## Hypothesis

Evoked-potential SNR metrics (max broadband SNR, tonotopic tuning) predict macro F1 consistently across different baseline architectures.

## Background

A frequency-decoding experiment was run on the same 41 sessions using 6 baseline architectures (EEGNet, GRU, Linear, MLP, ShallowCNN, TemporalCNN). This analysis tests whether the signal quality metrics developed in earlier RQs are consistent predictors of downstream decoding success.

## Method

Spearman rank correlations between SNR metrics and macro F1, computed separately per baseline (N = 41 sessions per baseline, 246 total observations). SNR metrics tested: Max Broadband SNR (ERP), Max Tonotopic Tuning, Mean Power Ratio SNR, Mean Induced Power SNR, Max Habituation Index.

- Script: `scripts/snr_baseline_correlation.py`

## Results

| Baseline | Max BB SNR ρ (p) | Max Tuning ρ (p) | Power Ratio ρ (p) | Induced ρ (p) | HI ρ (p) |
|----------|-----------------|------------------|-------------------|---------------|----------|
| EEGNet | **0.648** (4.6×10⁻⁶) | **0.665** (2.1×10⁻⁶) | 0.194 (0.223) | −0.148 (0.357) | 0.246 (0.120) |
| GRU | **0.592** (4.5×10⁻⁵) | **0.624** (1.3×10⁻⁵) | −0.040 (0.803) | −0.297 (0.060) | **0.330** (0.035) |
| ShallowCNN | **0.625** (1.2×10⁻⁵) | **0.558** (1.5×10⁻⁴) | 0.127 (0.429) | −0.154 (0.337) | 0.154 (0.337) |
| Linear | **0.542** (2.5×10⁻⁴) | **0.517** (5.4×10⁻⁴) | 0.125 (0.435) | −0.203 (0.204) | −0.012 (0.942) |
| MLP | **0.512** (6.2×10⁻⁴) | **0.457** (2.7×10⁻³) | 0.266 (0.093) | 0.050 (0.758) | 0.004 (0.982) |
| TemporalCNN | 0.275 (0.082) | 0.292 (0.064) | 0.096 (0.550) | 0.024 (0.881) | −0.011 (0.944) |
| **ALL (N=246)** | **0.478** (2.0×10⁻¹⁵) | **0.480** (1.4×10⁻¹⁵) | 0.116 (0.069) | −0.107 (0.095) | 0.119 (0.062) |

## Interpretation

Both Max Broadband SNR and Max Tonotopic Tuning are strong, consistent predictors of decoding performance — significant in 5/6 baselines (ρ = 0.46–0.67). Power Ratio SNR does NOT predict decoding despite revealing stimulus-driven power in 44.7% of channels. This critical dissociation means detecting *any* stimulus-driven power is not the same as having *frequency-discriminative* neural responses that enable classification.

TemporalCNN is the exception where neither metric reaches significance (p = 0.06–0.08). Induced Power SNR and Habituation Index are also non-predictive.

## Verdict

**Outcome:** Strongly Supported — Evoked-potential SNR (both max channel SNR and tonotopic tuning) is a robust predictor of decoding success across architectures. Sessions with high SNR consistently decode well regardless of model. Power Ratio SNR does not predict performance.

## Artifacts

- [Full HTML Report](results/snr_analysis_report.html)
- [Baseline Correlation Table](results/baseline_correlation_table.csv)

## Links

- Parent: [[snr-analysis/_overview]]
- Depends on: [[rq1-channel-session-snr]], [[rq4-tonotopic-tuning]], [[rq6-power-ratio-snr]], [[rq7-habituation]]
- Leads to: [[session-consistency]]
