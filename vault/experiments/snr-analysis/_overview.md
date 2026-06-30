---
title: "SNR Analysis — Minipig iEEG Signal Quality"
type: experiment
status: completed
created: 2025-06-15
updated: 2025-06-30
tags:
  - overview
  - snr
  - signal-quality
  - minipig
  - ieeg
---

# iEEG Signal Quality & SNR Analysis

This experiment systematically evaluates iEEG signal quality across all available minipig recording sessions. The goal is to identify which channels carry genuine neural responses to acoustic stimulation and which are dominated by noise, then to determine whether signal quality predicts downstream decoding performance.

## Dataset

| Metric | Value |
|--------|-------|
| Subjects | 7 (sub-01 to sub-07) |
| Recording Sessions | 41 |
| Total ECoG Channels | 736 |
| Channels per Session | 2–32 |

## Preprocessing Pipeline

1. Load session data from processed `.h5` files via `torch_brain.data.Data`
2. Extract epochs: 0.5 s rest window (pre-stimulus) and 0.5 s stimulus window per trial; ECoG channels only
3. Baseline correction: subtract mean of rest window from both segments
4. Broadband filtering: 1–300 Hz 4th-order Butterworth bandpass + 50 Hz notch (zero-phase `filtfilt`)
5. High-gamma filtering: 70–150 Hz 4th-order Butterworth bandpass (zero-phase)

## Core Metric: Evoked-Potential SNR

Per channel *c*:

$$\text{SNR}_{\text{evoked}, c} = \frac{\text{Var}(\bar{S}_c(t))}{\frac{1}{N} \sum_{i=1}^{N} \text{Var}(R_{i,c}(t))}$$

- **Numerator** — ERP variance: trial-averaged stimulus waveform variance (phase-locked signal)
- **Denominator** — mean single-trial rest variance (noise floor)
- **Threshold:** SNR > 0.5 → Active; otherwise → Dead

## Code

- Script: `scripts/run_snr_analysis.py`
- Module: `auditorydecoding/analysis/snr.py`
- Correlation script: `scripts/snr_baseline_correlation.py`

## Research Questions

| ID | Hypothesis | Status | File |
|----|-----------|--------|------|
| RQ1 | Channel & session SNR baseline | Supported | [[rq1-channel-session-snr]] |
| RQ2 | High-gamma sensitivity check | Not Supported | [[rq2-high-gamma-sensitivity]] |
| RQ3 | Trial-by-trial reliability (responsive ratio) | Supported | [[rq3-responsive-ratio]] |
| RQ4 | Frequency specificity (tonotopic tuning) | Supported | [[rq4-tonotopic-tuning]] |
| RQ5 | Low-frequency filtering improvement | Partially Supported | [[rq5-low-freq-filtering]] |
| RQ6 | Power ratio SNR (phase-locking assumption) | Strongly Supported | [[rq6-power-ratio-snr]] |
| RQ6a | Induced power decomposition | Strongly Supported | [[rq6a-induced-power]] |
| RQ7 | Neural habituation | Partially Supported | [[rq7-habituation]] |
| — | SNR vs. decoding performance | Strongly Supported | [[snr-vs-decoding-performance]] |
| — | Session consistency across baselines | Strongly Supported | [[session-consistency]] |

## Key Conclusions

1. 97.7% of channels are "Dead" by evoked SNR, but 44.7% show stimulus-driven power (power ratio > 1.0) — the signal is present but not phase-locked
2. Evoked-potential SNR (max broadband SNR and tonotopic tuning) predicts decoding F1 across 5/6 architectures (ρ = 0.46–0.67)
3. Session quality dominates model choice (Kendall's W = 0.747 across 6 baselines)
4. Induced (non-phase-locked) power accounts for 93.4% of total single-trial stimulus variance
5. Low-frequency filtering (1–70 Hz) is a safe default: small SNR gain, zero channel loss
6. Neural habituation is real but secondary to non-phase-locking

## Artifacts

- [Full HTML Report](results/snr_analysis_report.html)
- [Session Summary Table](results/session_summary_table.csv)
- [Channel Quality Table](results/channel_quality_table.csv)
- [Baseline Correlation Table](results/baseline_correlation_table.csv)
