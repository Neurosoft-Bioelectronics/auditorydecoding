---
id: rq1
title: "Channel-Level vs. Session-Level SNR"
experiment: "[[snr-analysis/_overview]]"
status: completed
outcome: supported
created: 2025-06-15
updated: 2025-06-30
tags:
  - signal-quality
  - snr
  - channel-selection
depends_on: []
leads_to:
  - "[[rq2-high-gamma-sensitivity]]"
  - "[[rq3-responsive-ratio]]"
  - "[[rq4-tonotopic-tuning]]"
---

# RQ1: Channel-Level vs. Session-Level SNR

## Question

Can we establish a baseline-corrected SNR to identify dead channels and aggregate valid channels into a reliable session score?

## Hypothesis

Evoked-potential SNR flags dead channels; averaging active channels gives a reliable session score.

## Background

Most electrodes in intracranial EEG recordings may not be positioned in auditory-responsive cortex. Before any downstream analysis or decoding, we need a principled way to separate channels with genuine stimulus-driven responses from those dominated by noise. This is the foundational question for the entire SNR analysis — all subsequent research questions build on the channel classification established here.

## Method

Compute per-channel evoked-potential SNR (trial-averaged stimulus variance / mean single-trial rest variance). Channels with SNR > 0.5 are classified as Active; the rest as Dead. Session-level SNR is the mean across active channels.

- Script: `scripts/run_snr_analysis.py`
- Module: `auditorydecoding/analysis/snr.py`

## Results

| Metric | Value |
|--------|-------|
| Dead Channels | 719 (97.7%) |
| Active Channels | 17 (2.3%) |
| Median BB SNR (all) | 0.0198 |
| Mean Active SNR | 0.76 |
| Sessions with active channels | 5 / 41 (12.2%) |

**SNR Percentile Distribution:**

| p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|-----|-----|-----|-----|-----|-----|-----|
| 0.0015 | 0.0054 | 0.0198 | 0.0472 | 0.1498 | 0.3513 | 0.6154 |

**Sessions with Active Channels:**

| Session | Active / Total | Mean Active SNR | Mean Resp. Ratio |
|---------|---------------|-----------------|------------------|
| sub-03_ses-04 LHanest | 6 / 24 | 0.5932 | 0.6998 |
| sub-03_ses-04 RHanest | 3 / 32 | 0.5681 | 0.6856 |
| sub-04_ses-01 RH | 1 / 2 | 1.5404 | 0.7924 |
| sub-05_ses-01 LH | 1 / 18 | 0.5008 | 0.8036 |
| sub-07_ses-03 LH | 6 / 18 | 0.8933 | 0.7570 |

## Interpretation

The overwhelming majority of channels (97.7%) are classified as Dead at the 0.5 SNR threshold. This indicates that most electrodes do not reliably capture evoked neural activity above their noise floor when assessed with broadband evoked-potential SNR. The few active channels cluster within specific sessions and subjects (sub-03, sub-04, sub-07), suggesting signal quality is highly dependent on surgical placement and recording conditions.

## Verdict

**Outcome:** Supported — The SNR metric clearly separates 17 active channels from 719 dead channels. However, only 5/41 sessions have any active channels, limiting the utility of session-level aggregation.

## Artifacts

- [Full HTML Report](results/snr_analysis_report.html)
- [Channel Quality Table](results/channel_quality_table.csv)
- [Session Summary Table](results/session_summary_table.csv)

## Links

- Parent: [[snr-analysis/_overview]]
- Leads to: [[rq2-high-gamma-sensitivity]], [[rq3-responsive-ratio]], [[rq4-tonotopic-tuning]]
- Related: [[snr-vs-decoding-performance]]
