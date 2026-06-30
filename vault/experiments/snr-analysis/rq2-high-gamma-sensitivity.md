---
id: rq2
title: "High-Gamma Sensitivity Check"
experiment: "[[snr-analysis/_overview]]"
status: completed
outcome: not-supported
created: 2025-06-15
updated: 2025-06-30
tags:
  - signal-quality
  - snr
  - high-gamma
  - frequency-band
depends_on:
  - "[[rq1-channel-session-snr]]"
leads_to:
  - "[[rq5-low-freq-filtering]]"
---

# RQ2: High-Gamma Sensitivity Check

## Question

Does evaluating signal quality in the high-gamma band (70–150 Hz) isolate neural signals better than broadband data?

## Hypothesis

High-gamma filtering isolates neural signals better than broadband, yielding higher SNR for responsive channels.

## Background

High-gamma activity (70–150 Hz) is often considered a closer proxy for local neural spiking activity in human ECoG. If auditory responses in minipig iEEG are driven by high-gamma modulations, filtering to this band could improve the signal-to-noise ratio by removing low-frequency artifacts and drift.

## Method

Apply a 70–150 Hz 4th-order Butterworth bandpass (zero-phase) and recompute evoked-potential SNR per channel. Compare high-gamma SNR to broadband SNR for both active and dead channel groups.

- Script: `scripts/run_snr_analysis.py`
- Module: `auditorydecoding/analysis/snr.py`

## Results

| Metric | Value |
|--------|-------|
| Mean HG SNR (all channels) | 0.0024 |
| Median HG SNR (all channels) | 0.0010 |
| Max HG SNR | 0.0048 |
| Channels above 0.5 in HG | 0 |

**SNR Difference (HG minus BB):**

| Channel Group | Mean BB SNR | Mean HG SNR | Mean Difference |
|---------------|------------|------------|-----------------|
| Active (17) | 0.7598 | 0.0044 | −0.7403 |
| Dead (719) | 0.0480 | 0.0023 | −0.0460 |

Pearson correlation between BB and HG SNR among active channels: r = 0.585, p = 0.014.

## Interpretation

Contrary to the hypothesis, high-gamma filtering does not improve SNR. It collapses the SNR of all channels to near-zero values. No channel reaches the 0.5 threshold in the high-gamma band. The auditory evoked responses in these minipig recordings are carried primarily by low-frequency components (ERP morphology in the broadband signal), not by high-gamma power modulations.

While the relative ranking is preserved (r = 0.585), the absolute magnitudes are far too low for high-gamma SNR to serve as a useful quality metric in this dataset.

## Verdict

**Outcome:** Not Supported — High-gamma SNR is extremely low for all channels. Evoked responses are carried primarily by low-frequency components.

## Artifacts

- [Full HTML Report](results/snr_analysis_report.html)
- [Channel Quality Table](results/channel_quality_table.csv)

## Links

- Parent: [[snr-analysis/_overview]]
- Depends on: [[rq1-channel-session-snr]]
- Leads to: [[rq5-low-freq-filtering]]
