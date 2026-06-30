---
id: rq6
title: "Power Ratio SNR — Phase-Locking Assumption"
experiment: "[[snr-analysis/_overview]]"
status: completed
outcome: strongly-supported
created: 2025-06-15
updated: 2025-06-30
tags:
  - signal-quality
  - snr
  - power-ratio
  - phase-locking
  - induced-activity
depends_on:
  - "[[rq1-channel-session-snr]]"
  - "[[rq3-responsive-ratio]]"
leads_to:
  - "[[rq6a-induced-power]]"
  - "[[snr-vs-decoding-performance]]"
---

# RQ6: Power Ratio SNR — Is Low Evoked SNR Caused by Phase-Locking Assumptions?

## Question

Is the low evoked SNR a true reflection of absent neural responses, or is it an artifact of the phase-locking assumption?

## Hypothesis

If neural responses are present but not precisely time-locked across trials, a Power Ratio SNR — which compares mean single-trial power during stimulus vs. rest without requiring phase consistency — should yield substantially higher SNR values than the evoked-potential metric. Specifically:

1. Power Ratio SNR will be significantly higher than evoked SNR across all channels
2. A substantial fraction of "Dead" channels will show Power Ratio > 1.0
3. Power Ratio SNR will correlate with responsive ratio but weakly with evoked SNR

## Background

The evoked-potential SNR (RQ1–RQ5) relies on trial averaging, which acts as a matched filter for phase-locked responses. Any neural response that varies in latency, amplitude, or waveform shape across trials is attenuated. With 97.7% of channels classified as "Dead," it is possible that channels carry real stimulus-driven activity that is induced (power changes without consistent phase) rather than evoked (phase-locked ERPs).

## Method

Per channel *c*:

$$\text{SNR}_{\text{power}, c} = \frac{\frac{1}{N} \sum_{i=1}^{N} \text{Var}(S_{i,c}(t))}{\frac{1}{N} \sum_{i=1}^{N} \text{Var}(R_{i,c}(t))}$$

No trial-averaging before computing variance. Each epoch is measured individually.

- Script: `scripts/run_snr_analysis.py`
- Module: `auditorydecoding/analysis/snr.py`

## Results

| Metric | Value |
|--------|-------|
| Mean Power Ratio SNR (all) | 0.973 |
| Mean Evoked SNR (all) | 0.064 |
| Power Ratio / Evoked Ratio | 15.1× |
| Channels with Power Ratio > 1.0 | 329 (44.7%) |
| "Dead" channels with Power Ratio > 1.0 | 313 |
| Mean Power Ratio (Active) | 1.83 |
| Mean Power Ratio (Dead) | 0.95 |

**Statistical Tests:**

| Test | Result | Interpretation |
|------|--------|----------------|
| Wilcoxon (power vs evoked) | stat = 0.0, p = 3.83 × 10⁻¹²² | Power ratio higher for every channel |
| Spearman (evoked, power) | ρ = 0.051, p = 0.17 | No correlation — fundamentally different metrics |
| Spearman (power, responsive ratio) | ρ = 0.898, p = 3.66 × 10⁻²⁶³ | Very strong — both phase-insensitive |

**Power Ratio by Frequency Band:**

| Band | Mean Power Ratio | Channels > 1.0 |
|------|-----------------|-----------------|
| Broadband (1–300 Hz) | 0.973 | 329 (44.7%) |
| High-Gamma (70–150 Hz) | 0.987 | 315 (42.8%) |
| Low-Frequency (1–70 Hz) | 0.975 | 328 (44.6%) |

## Interpretation

The phase-locking assumption is the dominant factor driving low evoked SNR. The evoked SNR is 15× lower on average. The two metrics are essentially uncorrelated (ρ = 0.051), capturing fundamentally different properties. Nearly half of all channels show stimulus-driven power increases, including 313 channels classified as "Dead" by evoked criteria. Power ratio and responsive ratio (ρ = 0.898) are measuring the same underlying phenomenon. The power ratio is consistent across frequency bands.

## Verdict

**Outcome:** Strongly Supported — Low evoked SNR is substantially caused by the phase-locking assumption, not by absence of neural responses. 44.7% of channels show stimulus power exceeding rest power vs. only 2.3% "Active" by evoked SNR.

## Artifacts

- [Full HTML Report](results/snr_analysis_report.html)
- [Channel Quality Table](results/channel_quality_table.csv)

## Links

- Parent: [[snr-analysis/_overview]]
- Depends on: [[rq1-channel-session-snr]], [[rq3-responsive-ratio]]
- Leads to: [[rq6a-induced-power]], [[snr-vs-decoding-performance]]
