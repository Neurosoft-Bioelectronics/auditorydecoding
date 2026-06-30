---
id: rq4
title: "Frequency Specificity (Tonotopic Tuning)"
experiment: "[[snr-analysis/_overview]]"
status: completed
outcome: supported
created: 2025-06-15
updated: 2025-06-30
tags:
  - signal-quality
  - tonotopic-tuning
  - frequency-specificity
depends_on:
  - "[[rq1-channel-session-snr]]"
leads_to:
  - "[[snr-vs-decoding-performance]]"
---

# RQ4: Frequency Specificity (Tonotopic Tuning)

## Question

Does the SNR vary per channel depending on sound frequency, indicating true tonotopic neural tuning?

## Hypothesis

SNR varies by stimulus frequency on responsive channels, reflecting genuine tonotopic neural processing.

## Background

Tonotopic organization — where different neural populations respond preferentially to different sound frequencies — is a hallmark of auditory cortex. If active channels show frequency-specific SNR variation, this confirms they are recording from auditory-responsive cortex rather than picking up non-specific artifacts.

## Method

Compute per-frequency SNR for each channel across stimulus types. The tuning metric is the standard deviation of per-frequency SNR values within each channel. Compare tuning between Active and Dead groups. Test correlation with broadband SNR.

- Script: `scripts/run_snr_analysis.py`
- Module: `auditorydecoding/analysis/snr.py`

## Results

| Metric | Active Channels | Dead Channels |
|--------|----------------|---------------|
| Mean Tuning | 0.837 | 0.090 |
| Max Tuning | 3.43 | 1.36 |

Standout channel: CG2D2E1 in sub-04_ses-01 RH (tuning = 3.43, BB SNR = 1.54).

Pearson correlation between BB SNR and tuning (active channels): r = 0.869, p < 0.001.

## Interpretation

Active channels exhibit strong frequency-dependent SNR variation (~10x higher tuning than dead channels), consistent with genuine tonotopic neural processing. Dead channels show flat, near-zero tuning — their SNR does not meaningfully change with stimulus frequency. The tight correlation between SNR and tuning (r = 0.87) suggests channels with better signal quality also capture more frequency-specific neural information.

## Verdict

**Outcome:** Supported — Active channels show ~10x higher tuning metrics than dead channels, with strong frequency-specific responses.

## Artifacts

- [Full HTML Report](results/snr_analysis_report.html)
- [Channel Quality Table](results/channel_quality_table.csv)

## Links

- Parent: [[snr-analysis/_overview]]
- Depends on: [[rq1-channel-session-snr]]
- Leads to: [[snr-vs-decoding-performance]]
