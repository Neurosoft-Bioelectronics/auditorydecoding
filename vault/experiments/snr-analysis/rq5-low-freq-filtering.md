---
id: rq5
title: "Low-Frequency Filtering (1–70 Hz) vs. Broadband"
experiment: "[[snr-analysis/_overview]]"
status: completed
outcome: partially-supported
created: 2025-06-15
updated: 2025-06-30
tags:
  - signal-quality
  - snr
  - filtering
  - low-frequency
depends_on:
  - "[[rq1-channel-session-snr]]"
  - "[[rq2-high-gamma-sensitivity]]"
leads_to: []
---

# RQ5: Low-Frequency Filtering (1–70 Hz) vs. Broadband

## Question

Does low-frequency filtering (1–70 Hz) improve SNR by removing high-frequency noise while preserving the dominant low-frequency evoked potentials?

## Hypothesis

Low-frequency filtering (1–70 Hz + 50 Hz notch) will yield higher channel-level SNR and responsive ratios compared to broadband filtering (1–300 Hz + 50 Hz notch), because high-frequency noise is removed while the dominant low-frequency evoked potentials are preserved.

## Background

The high-gamma analysis ([[rq2-high-gamma-sensitivity]]) revealed near-zero SNR above 70 Hz, indicating that frequencies above 70 Hz contribute mainly noise in these recordings. Restricting the passband to 1–70 Hz should remove noise-dominated frequencies while preserving the evoked potentials.

## Method

Recompute all SNR metrics with a 1–70 Hz + 50 Hz notch filter. Compare to broadband (1–300 Hz) at the channel level and session level. Test with Wilcoxon signed-rank test and Spearman rank correlation.

- Script: `scripts/run_snr_analysis.py`
- Module: `auditorydecoding/analysis/snr.py`

## Results

| Metric | Broadband (1–300 Hz) | Low-Freq (1–70 Hz) | Difference |
|--------|---------------------|--------------------|-----------:|
| Active channels | 17 (2.3%) | 20 (2.7%) | +3 |
| Mean SNR (all) | 0.0644 | 0.0659 | +0.0015 |
| Median SNR (all) | 0.0198 | 0.0209 | +0.0012 |
| Mean SNR (BB-active) | 0.7450 | 0.7218 | −0.0232 |
| Mean resp. ratio (all) | 0.4755 | 0.4766 | +0.0011 |
| Mean tuning (BB-active) | 0.8367 | 0.8998 | +0.0631 |

- Wilcoxon signed-rank: p = 1.06 × 10⁻⁵² (LF > BB for 78.9% of channels)
- Spearman rank correlation (BB vs LF SNR): ρ = 0.992
- Three newly activated channels: two in sub-05_ses-01, one borderline in sub-03_ses-04
- Zero channels lost

## Interpretation

Low-frequency filtering provides a statistically significant but practically small SNR improvement. The Wilcoxon test confirms 78.9% of channels improve, but the absolute gains are modest (+0.0015 mean). Three additional channels cross the 0.5 threshold with zero channels lost. Tonotopic tuning is marginally enhanced (+7.5%). The two filter bands produce near-identical channel rankings (ρ = 0.992).

## Verdict

**Outcome:** Partially Supported — Statistically significant but small SNR improvement. Three channels recovered, zero lost. Low-frequency filtering is a safe default with modest but cost-free gains.

## Artifacts

- [Full HTML Report](results/snr_analysis_report.html)
- [Channel Quality Table](results/channel_quality_table.csv)

## Links

- Parent: [[snr-analysis/_overview]]
- Depends on: [[rq1-channel-session-snr]], [[rq2-high-gamma-sensitivity]]
