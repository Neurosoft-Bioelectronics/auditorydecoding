---
id: rq7
title: "Neural Habituation and ERP SNR"
experiment: "[[snr-analysis/_overview]]"
status: completed
outcome: partially-supported
created: 2025-06-15
updated: 2025-06-30
tags:
  - signal-quality
  - snr
  - habituation
  - temporal-dynamics
depends_on:
  - "[[rq1-channel-session-snr]]"
  - "[[rq6a-induced-power]]"
leads_to:
  - "[[snr-vs-decoding-performance]]"
---

# RQ7: Does Neural Habituation Explain Low ERP SNR?

## Question

Do early trials produce strong phase-locked ERPs that habituate over the session, diluting the overall evoked SNR when all trials are averaged?

## Hypothesis

If neural habituation is a significant factor (Hypothesis B):

1. Evoked SNR from early trials (Q1) should be substantially higher than SNR from late trials (Q4)
2. The Habituation Index should be positive for most channels
3. Cumulative ERP SNR should peak early and decline as more trials are added

## Background

An alternative explanation for low evoked SNR (complementary to Hypothesis A from [[rq6-power-ratio-snr]]): early trials may produce strong phase-locked ERPs that diminish over the session. If habituation is substantial, averaging across all trials would dilute strong early responses with flat late responses.

## Method

**Quartile SNR:** Split trials chronologically into Q1–Q4, compute evoked SNR per quartile.

**Habituation Index (HI):**

$$\text{HI}_c = \frac{\text{SNR}_{\text{early},c} - \text{SNR}_{\text{late},c}}{\text{SNR}_{\text{early},c} + \text{SNR}_{\text{late},c} + \epsilon}$$

- HI > 0 → habituation; HI < 0 → sensitization; HI ≈ 0 → stable

**Cumulative ERP SNR:** Compute ERP SNR using first K trials for K = 5, 10, 15, ..., N.

- Script: `scripts/run_snr_analysis.py`
- Module: `auditorydecoding/analysis/snr.py`

## Results

| Metric | Value |
|--------|-------|
| Mean Habituation Index | 0.200 |
| Median Habituation Index | 0.249 |
| Mean Q1 SNR (earliest 25%) | 0.103 |
| Mean Q4 SNR (latest 25%) | 0.065 |
| Channels with HI > 0.1 (habituation) | 461 (62.6%) |
| Channels with HI < −0.1 (sensitization) | 194 (26.4%) |
| Wilcoxon Q1 vs Q4 | stat = 73,523, p = 5.31 × 10⁻²⁷ |
| Spearman (HI, Evoked SNR) | ρ = 0.238, p = 6.06 × 10⁻¹¹ |

## Interpretation

Both mechanisms contribute. The median HI of 0.249 and highly significant Q1 vs Q4 difference confirm neural habituation is real and measurable — early trials produce ~59% higher evoked SNR than late trials (0.103 vs 0.065). However, even Q1 SNR (0.103) remains far below the 0.5 "Active" threshold, indicating habituation alone does not explain the low evoked SNR. Most of the deficit comes from non-phase-locking (Hypothesis A, confirmed in [[rq6a-induced-power]]).

The 62.6% of channels with HI > 0.1 shows systematic response diminishment, partially offset by 26.4% showing sensitization. The weak positive correlation between HI and evoked SNR (ρ = 0.238) suggests the best-responding channels are also the most prone to habituation.

## Verdict

**Outcome:** Partially Supported — Median HI = 0.249; Q1 SNR is 59% higher than Q4 (p = 5.31 × 10⁻²⁷). 62.6% of channels habituate. But even Q1 SNR remains below Active threshold, so habituation is secondary to non-phase-locking.

## Artifacts

- [Full HTML Report](results/snr_analysis_report.html)

## Links

- Parent: [[snr-analysis/_overview]]
- Depends on: [[rq1-channel-session-snr]], [[rq6a-induced-power]]
- Leads to: [[snr-vs-decoding-performance]]
